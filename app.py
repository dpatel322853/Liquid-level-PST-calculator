"""
Vessel Inventory and PST/TTC Screening Calculator
=================================================

Preliminary engineering screening only. This application is not a substitute
for approved project calculations, dynamic simulation, HAZOP/LOPA, SRS, alarm
rationalization, SIS design, operating procedures, or engineering approval.

Run:
    pip install -r requirements.txt
    streamlit run app_corrected.py

Manual validation cases
-----------------------
1. Empty vessel: volume_at_level(0) = 0.
2. Full flat-head vertical cylinder: V = pi*D^2/4*H.
3. Half-full flat-head horizontal cylinder: V = 0.5*pi*D^2/4*L.
4. High-level PST: use V(high hazard) - V(effective HHLL), never
   V(effective HHLL) - V(current).
5. Low-level PST: use V(effective LLLL) - V(low hazard), never
   V(current) - V(effective LLLL).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

EPS = 1.0e-12
HEAD_TYPES = [
    "Flat",
    "2:1 ellipsoidal",
    "Hemispherical",
    "Torispherical (approx.)",
    "Conical (approx.)",
]
EQUIPMENT_TYPES = [
    "Vertical cylindrical vessel",
    "Horizontal cylindrical vessel",
    "Horizontal vessel with heads",
    "Vertical vessel with heads",
    "Kettle exchanger / reboiler",
]


@dataclass(frozen=True)
class Geometry:
    equipment_type: str
    diameter_m: float
    straight_length_m: float
    head_type: str = "Flat"
    number_of_heads: int = 0
    cone_height_m: float = 0.0
    bundle_displacement_m3: float = 0.0
    subtract_bundle: bool = False

    @property
    def is_vertical(self) -> bool:
        return self.equipment_type.startswith("Vertical")

    @property
    def is_kettle(self) -> bool:
        return self.equipment_type.startswith("Kettle")

    @property
    def maximum_level_m(self) -> float:
        if self.is_vertical:
            return self.straight_length_m + self.number_of_heads * head_depth_m(
                self.diameter_m, self.head_type, self.cone_height_m
            )
        return self.diameter_m


# =============================================================================
# Geometry and inventory functions
# =============================================================================
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def horizontal_cylinder_segment_area(
    diameter_m: float, liquid_height_m: float
) -> float:
    """Exact circular segment area for a horizontal cylinder, m2."""
    radius = diameter_m / 2.0
    height = clamp(liquid_height_m, 0.0, diameter_m)
    if height <= 0.0:
        return 0.0
    if height >= diameter_m:
        return math.pi * radius**2
    root_term = max(0.0, 2.0 * radius * height - height**2)
    return (
        radius**2 * math.acos((radius - height) / radius)
        - (radius - height) * math.sqrt(root_term)
    )


def horizontal_cylinder_liquid_volume(
    diameter_m: float, length_m: float, liquid_height_m: float
) -> float:
    return horizontal_cylinder_segment_area(diameter_m, liquid_height_m) * length_m


def vertical_cylinder_liquid_volume(
    diameter_m: float, liquid_height_m: float
) -> float:
    area = math.pi * diameter_m**2 / 4.0
    return area * max(0.0, liquid_height_m)


def head_depth_m(
    diameter_m: float, head_type: str, cone_height_m: float = 0.0
) -> float:
    if head_type == "Flat":
        return 0.0
    if head_type == "2:1 ellipsoidal":
        return diameter_m / 4.0
    if head_type == "Hemispherical":
        return diameter_m / 2.0
    if head_type == "Torispherical (approx.)":
        # Approximate inside depth for a standard flanged-and-dished head.
        return 0.1935 * diameter_m
    if head_type == "Conical (approx.)":
        return cone_height_m if cone_height_m > 0.0 else diameter_m / 4.0
    return 0.0


def head_volume_estimate(
    diameter_m: float, head_type: str, cone_height_m: float = 0.0
) -> float:
    """Full internal volume of one head, m3.

    Ellipsoidal, hemispherical, and conical values are geometric estimates.
    Torispherical volume is a clearly labelled preliminary approximation.
    """
    radius = diameter_m / 2.0
    if head_type == "Flat":
        return 0.0
    if head_type == "2:1 ellipsoidal":
        semi_axis = diameter_m / 4.0
        return (2.0 / 3.0) * math.pi * radius**2 * semi_axis
    if head_type == "Hemispherical":
        return (2.0 / 3.0) * math.pi * radius**3
    if head_type == "Torispherical (approx.)":
        return 0.084 * diameter_m**3
    if head_type == "Conical (approx.)":
        return math.pi * radius**2 * head_depth_m(
            diameter_m, head_type, cone_height_m
        ) / 3.0
    return 0.0


def vertical_bottom_head_partial_volume(
    diameter_m: float,
    head_type: str,
    height_in_head_m: float,
    cone_height_m: float = 0.0,
) -> float:
    """Volume filled from the bottom apex of one vertical head.

    Ellipsoidal and hemispherical heads use the exact normalized half-ellipsoid
    slice relationship. Conical heads use the exact similar-triangle relation.
    Torispherical heads use a smooth monotonic screening approximation.
    """
    depth = head_depth_m(diameter_m, head_type, cone_height_m)
    full_volume = head_volume_estimate(diameter_m, head_type, cone_height_m)
    if depth <= EPS:
        return 0.0
    height = clamp(height_in_head_m, 0.0, depth)
    fraction = height / depth
    if head_type == "Conical (approx.)":
        normalized_volume = fraction**3
    elif head_type in {"2:1 ellipsoidal", "Hemispherical"}:
        normalized_volume = 1.5 * fraction**2 - 0.5 * fraction**3
    else:
        normalized_volume = 3.0 * fraction**2 - 2.0 * fraction**3
    return full_volume * normalized_volume


def horizontal_head_partial_volume(
    diameter_m: float,
    liquid_height_m: float,
    full_head_volume_m3: float,
) -> float:
    """Screening approximation for a partially filled horizontal head.

    The head is assigned the same filled-area fraction as the adjacent circular
    shell cross-section. This is not an exact integration of head geometry.
    """
    full_area = math.pi * diameter_m**2 / 4.0
    if full_area <= EPS:
        return 0.0
    fill_fraction = (
        horizontal_cylinder_segment_area(diameter_m, liquid_height_m) / full_area
    )
    return full_head_volume_m3 * fill_fraction


def vessel_volume_at_level(
    geometry: Geometry, liquid_level_m: float, gross: bool = False
) -> float:
    """Return gross or net liquid volume at level using SI units."""
    diameter = geometry.diameter_m
    length = geometry.straight_length_m
    level = clamp(liquid_level_m, 0.0, geometry.maximum_level_m)
    one_head_volume = head_volume_estimate(
        diameter, geometry.head_type, geometry.cone_height_m
    )

    if geometry.is_vertical:
        head_depth = head_depth_m(
            diameter, geometry.head_type, geometry.cone_height_m
        )
        has_bottom_head = geometry.number_of_heads >= 1
        has_top_head = geometry.number_of_heads >= 2

        if has_bottom_head and level < head_depth:
            volume = vertical_bottom_head_partial_volume(
                diameter, geometry.head_type, level, geometry.cone_height_m
            )
        else:
            volume = one_head_volume if has_bottom_head else 0.0
            shell_level_origin = head_depth if has_bottom_head else 0.0
            shell_height = clamp(level - shell_level_origin, 0.0, length)
            volume += vertical_cylinder_liquid_volume(diameter, shell_height)

            if has_top_head and level > shell_level_origin + length:
                penetration = clamp(
                    level - shell_level_origin - length, 0.0, head_depth
                )
                empty_cap = vertical_bottom_head_partial_volume(
                    diameter,
                    geometry.head_type,
                    head_depth - penetration,
                    geometry.cone_height_m,
                )
                volume += one_head_volume - empty_cap
    else:
        volume = horizontal_cylinder_liquid_volume(diameter, length, level)
        if geometry.number_of_heads > 0:
            volume += geometry.number_of_heads * horizontal_head_partial_volume(
                diameter, level, one_head_volume
            )

    if geometry.is_kettle and geometry.subtract_bundle and not gross:
        # Uniform level-fraction allocation is a screening approximation.
        allocated_displacement = geometry.bundle_displacement_m3 * clamp(
            level / diameter, 0.0, 1.0
        )
        volume -= allocated_displacement

    return max(0.0, volume)


def kettle_liquid_volume(
    geometry: Geometry, liquid_level_m: float
) -> Tuple[float, float]:
    gross = vessel_volume_at_level(geometry, liquid_level_m, gross=True)
    net = vessel_volume_at_level(geometry, liquid_level_m, gross=False)
    return gross, net


def total_internal_volume(geometry: Geometry, gross: bool = False) -> float:
    return vessel_volume_at_level(geometry, geometry.maximum_level_m, gross=gross)


# =============================================================================
# PST, setpoint tolerance, and IRT functions
# =============================================================================
def apply_setpoint_tolerance(
    nominal_setpoint_m: float,
    scenario: str,
    tolerance_value: float,
    tolerance_basis: str,
    transmitter_range_m: float,
    vessel_level_span_m: float,
) -> Tuple[float, float, bool]:
    """Return effective setpoint, absolute shift, and whether clamping occurred.

    The convention required by the calculation specification is implemented:
    high-level effective HHLL = nominal HHLL - tolerance;
    low-level effective LLLL = nominal LLLL + tolerance.
    """
    if tolerance_basis == "Absolute level unit":
        shift_m = tolerance_value
    elif tolerance_basis == "Percent of transmitter calibrated range":
        shift_m = tolerance_value / 100.0 * transmitter_range_m
    else:
        shift_m = tolerance_value / 100.0 * vessel_level_span_m

    unbounded = (
        nominal_setpoint_m - shift_m
        if scenario == "High Level"
        else nominal_setpoint_m + shift_m
    )
    effective = clamp(unbounded, 0.0, vessel_level_span_m)
    return effective, shift_m, not math.isclose(unbounded, effective, abs_tol=EPS)


def calculate_time_seconds(
    inventory_change_m3: float, directional_flow_m3_h: float
) -> Optional[float]:
    if inventory_change_m3 < -EPS or directional_flow_m3_h <= EPS:
        return None
    return max(0.0, inventory_change_m3) / directional_flow_m3_h * 3600.0


def calculate_time_to_trip(
    current_volume_m3: float,
    effective_trip_volume_m3: float,
    directional_flow_m3_h: float,
    scenario: str,
) -> Optional[float]:
    """Current level to effective trip. This function never calculates PST."""
    inventory_change = (
        effective_trip_volume_m3 - current_volume_m3
        if scenario == "High Level"
        else current_volume_m3 - effective_trip_volume_m3
    )
    return calculate_time_seconds(inventory_change, directional_flow_m3_h)


def calculate_pst_time_to_consequence(
    effective_trip_volume_m3: float,
    hazard_volume_m3: float,
    directional_flow_m3_h: float,
    scenario: str,
) -> Optional[float]:
    """Effective trip to hazard endpoint. Current level is intentionally absent."""
    inventory_change = (
        hazard_volume_m3 - effective_trip_volume_m3
        if scenario == "High Level"
        else effective_trip_volume_m3 - hazard_volume_m3
    )
    return calculate_time_seconds(inventory_change, directional_flow_m3_h)


def calculate_total_time_to_hazard(
    current_volume_m3: float,
    hazard_volume_m3: float,
    directional_flow_m3_h: float,
    scenario: str,
) -> Optional[float]:
    inventory_change = (
        hazard_volume_m3 - current_volume_m3
        if scenario == "High Level"
        else current_volume_m3 - hazard_volume_m3
    )
    return calculate_time_seconds(inventory_change, directional_flow_m3_h)


def calculate_irt(
    sensor_response_s: float,
    logic_solver_s: float,
    final_element_s: float,
    additional_lag_s: float,
    manual_override_s: Optional[float] = None,
) -> float:
    """IRT is calculated separately from all process inventory times."""
    if manual_override_s is not None:
        return manual_override_s
    return (
        sensor_response_s
        + logic_solver_s
        + final_element_s
        + additional_lag_s
    )


def evaluate_pst_vs_irt(
    pst_s: Optional[float], irt_s: float
) -> Dict[str, object]:
    if pst_s is None or pst_s <= EPS:
        return {
            "status": "NOT EVALUATED",
            "pst_irt_ratio": None,
            "irt_pst_percent": None,
            "pst_gt_irt": False,
            "preferred": False,
        }
    ratio = math.inf if irt_s <= EPS else pst_s / irt_s
    percentage = 100.0 * irt_s / pst_s
    if irt_s >= pst_s:
        status = "FAIL"
    elif irt_s <= pst_s / 2.0:
        status = "PREFERRED"
    else:
        status = "WARNING"
    return {
        "status": status,
        "pst_irt_ratio": ratio,
        "irt_pst_percent": percentage,
        "pst_gt_irt": irt_s < pst_s,
        "preferred": irt_s <= pst_s / 2.0,
    }


def seconds_text(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "N/A"
    return (
        f"{seconds:,.1f} s | {seconds / 60.0:,.2f} min | "
        f"{seconds / 3600.0:,.3f} h"
    )


def build_pst_summary(data: Dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([data])


def length_to_m(value: float, unit_system: str) -> float:
    return value / 1000.0 if unit_system.startswith("Engineering") else value


def level_for_display(value_m: float, unit_system: str) -> float:
    return value_m * 1000.0 if unit_system.startswith("Engineering") else value_m


def flow_to_m3_h(value: float, flow_unit: str, density_kg_m3: float) -> float:
    if flow_unit == "kg/h":
        return value / density_kg_m3 if density_kg_m3 > EPS else 0.0
    return value


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "pst_summary"


# =============================================================================
# Streamlit interface
# =============================================================================
st.set_page_config(
    page_title="Vessel Inventory and PST/TTC Calculator",
    page_icon="🛡️",
    layout="wide",
)
st.title("Vessel Inventory and PST / TTC Screening Calculator")
st.caption(
    "Preliminary vessel inventory, Time to Trip, Process Safety Time / Time "
    "to Consequence, and Instrument Response Time screening."
)

with st.expander(
    "⚠️ Safety, confidentiality, and intended-use disclaimer", expanded=True
):
    st.warning(
        "PRELIMINARY ENGINEERING SCREENING ONLY. Do not use these results "
        "directly for final design, SIS design, alarm rationalization, HAZOP, "
        "LOPA, SRS approval, operating procedures, or any safety-critical "
        "decision. Verification by qualified Process, Process Safety, and "
        "Control Systems engineers is mandatory. Use only in an information "
        "environment approved for the applicable project confidentiality."
    )

with st.sidebar:
    st.header("Calculation Setup")
    unit_system = st.selectbox(
        "Unit system",
        ["SI (m, m3, m3/h)", "Engineering (mm, m3, m3/h)"],
    )
    level_unit = "mm" if unit_system.startswith("Engineering") else "m"
    scenario = st.radio("Scenario", ["High Level", "Low Level"])
    st.caption("The calculation engine uses SI units internally.")

with st.form("pst_inputs"):
    geometry_tab, level_tab, flow_tab, basis_tab, irt_tab, srs_tab = st.tabs(
        [
            "Equipment Geometry",
            "Level Setpoints",
            "Flow Basis",
            "PST/TTC Basis",
            "IRT Basis",
            "SRS Documentation",
        ]
    )

    with geometry_tab:
        g1, g2, g3 = st.columns(3)
        equipment_tag = g1.text_input("Equipment tag/name", "V-1001")
        equipment_type = g2.selectbox("Equipment type", EQUIPMENT_TYPES)
        diameter_u = g3.number_input(
            f"Vessel/shell internal diameter ({level_unit})",
            min_value=0.001,
            value=2.0 if level_unit == "m" else 2000.0,
        )
        straight_length_u = g1.number_input(
            f"Straight-side or tangent-to-tangent length ({level_unit})",
            min_value=0.001,
            value=5.0 if level_unit == "m" else 5000.0,
        )
        heads_applicable = (
            "with heads" in equipment_type.lower()
            or equipment_type.startswith("Kettle")
        )
        head_type = g2.selectbox(
            "Head type", HEAD_TYPES, disabled=not heads_applicable
        )
        number_of_heads = g3.selectbox(
            "Number of heads",
            [0, 1, 2],
            index=2 if heads_applicable else 0,
            disabled=not heads_applicable,
        )
        cone_height_u = g1.number_input(
            f"Conical-head axial depth ({level_unit})",
            min_value=0.0,
            value=0.0,
            disabled=(
                not heads_applicable or head_type != "Conical (approx.)"
            ),
        )
        bundle_displacement_m3 = g2.number_input(
            "Tube-bundle displacement (m3)",
            min_value=0.0,
            value=0.0,
            disabled=not equipment_type.startswith("Kettle"),
        )
        subtract_bundle = g3.checkbox(
            "Subtract tube-bundle displacement",
            value=True,
            disabled=not equipment_type.startswith("Kettle"),
        )

    with level_tab:
        st.caption(
            f"All levels are vertical distances from the equipment bottom in {level_unit}."
        )
        l1, l2, l3 = st.columns(3)
        current_u = l1.number_input(
            f"Current liquid level ({level_unit})",
            min_value=0.0,
            value=1.0 if level_unit == "m" else 1000.0,
        )
        llll_u = l2.number_input(
            f"LLLL trip setpoint ({level_unit})",
            min_value=0.0,
            value=0.30 if level_unit == "m" else 300.0,
        )
        ll_u = l3.number_input(
            f"Low alarm LL/L ({level_unit})",
            min_value=0.0,
            value=0.50 if level_unit == "m" else 500.0,
        )
        high_alarm_u = l1.number_input(
            f"High alarm H/HLL ({level_unit})",
            min_value=0.0,
            value=1.50 if level_unit == "m" else 1500.0,
        )
        hhll_u = l2.number_input(
            f"HHLL trip setpoint ({level_unit})",
            min_value=0.0,
            value=1.70 if level_unit == "m" else 1700.0,
        )
        high_hazard_u = l3.number_input(
            f"High-level hazard endpoint ({level_unit})",
            min_value=0.0,
            value=1.90 if level_unit == "m" else 1900.0,
        )
        low_hazard_u = l1.number_input(
            f"Low-level hazard endpoint ({level_unit})",
            min_value=0.0,
            value=0.10 if level_unit == "m" else 100.0,
        )
        tolerance_basis = l2.selectbox(
            "Setpoint tolerance basis",
            [
                "Percent of transmitter calibrated range",
                "Percent of vessel level span",
                "Absolute level unit",
            ],
        )
        default_tolerance = (
            2.0
            if tolerance_basis != "Absolute level unit"
            else (0.02 if level_unit == "m" else 20.0)
        )
        hhll_tolerance_u = l3.number_input(
            "HHLL tolerance (% or absolute)",
            min_value=0.0,
            value=default_tolerance,
        )
        llll_tolerance_u = l1.number_input(
            "LLLL tolerance (% or absolute)",
            min_value=0.0,
            value=default_tolerance,
        )
        transmitter_span_u = l2.number_input(
            f"Transmitter calibrated range/span ({level_unit})",
            min_value=0.001,
            value=2.0 if level_unit == "m" else 2000.0,
        )
        st.info(
            "The default percentage tolerance is 2%. Replace it with the "
            "approved project or Control Systems value. Effective HHLL is "
            "shifted lower and effective LLLL higher as required by this methodology."
        )

    with flow_tab:
        f1, f2, f3 = st.columns(3)
        flow_unit = f1.selectbox("Input flow unit", ["m3/h", "kg/h"])
        density = f2.number_input(
            "Liquid density (kg/m3)",
            min_value=0.001,
            value=850.0,
            disabled=flow_unit != "kg/h",
        )
        pst_flow_basis = f3.radio(
            "PST/TTC flow basis", ["Worst-case rates", "Normal rates"]
        )
        normal_inflow_u = f1.number_input(
            f"Normal inflow ({flow_unit})", min_value=0.0, value=100.0
        )
        normal_outflow_u = f2.number_input(
            f"Normal outflow ({flow_unit})", min_value=0.0, value=60.0
        )
        worst_inflow_u = f1.number_input(
            f"Worst-case inflow ({flow_unit})", min_value=0.0, value=120.0
        )
        worst_outflow_u = f2.number_input(
            f"Worst-case outflow ({flow_unit})", min_value=0.0, value=20.0
        )
        st.caption(
            "Time to Trip and Total Time to Hazard use normal net flow. PST/TTC "
            "uses the explicitly selected normal or worst-case flow basis."
        )

    with basis_tab:
        b1, b2 = st.columns(2)
        initiating_variable = b1.selectbox(
            "Initiating variable",
            ["HHLL", "LLLL", "Other"],
            index=0 if scenario == "High Level" else 1,
        )
        operating_mode = b2.selectbox(
            "Applicable plant mode",
            [
                "Startup", "Normal operation", "Turndown", "Shutdown",
                "Recycle", "Maintenance", "Bypass", "Other",
            ],
            index=1,
        )
        calculation_basis = b1.text_area(
            "Calculation basis",
            "Steady net liquid flow and static vessel geometry; preliminary screening basis.",
        )
        data_source = b2.text_area(
            "Data source",
            "Equipment datasheet, P&ID, H&MB, setpoint summary; verify approved revisions.",
        )
        user_limitations = st.text_area(
            "Additional assumptions and limitations",
            "No flashing, foaming, entrainment, level swell, gas compression, or changing flow with level.",
        )

    with irt_tab:
        i1, i2, i3 = st.columns(3)
        irt_method = i1.radio(
            "IRT method", ["Preliminary calculation", "Manual IRT"]
        )
        signal_type = i2.selectbox(
            "Sensor signal type",
            ["Level", "Pressure", "Flow", "Temperature thermowell", "Other"],
        )
        sensor_default = 30.0 if signal_type == "Temperature thermowell" else 1.0
        sensor_response_s = i3.number_input(
            "Sensor response time (s)", min_value=0.0, value=sensor_default
        )
        logic_solver_s = i1.number_input(
            "Logic solver time (s)", min_value=0.0, value=1.0
        )
        final_element_type = i2.selectbox(
            "Final element type",
            ["Shutdown valve", "Control valve", "Pump trip", "Compressor trip", "Other"],
        )
        valve_size_in = i3.number_input(
            "Shutdown valve size (inch)",
            min_value=0.0,
            value=6.0,
            disabled=final_element_type != "Shutdown valve",
        )
        valve_time_basis = i1.selectbox(
            "Shutdown valve travel-time basis",
            ["1 second per inch", "Manual entry"],
            disabled=final_element_type != "Shutdown valve",
        )
        manual_final_element_s = i2.number_input(
            "Manual final-element response time (s)",
            min_value=0.0,
            value=5.0,
            disabled=(
                final_element_type == "Shutdown valve"
                and valve_time_basis == "1 second per inch"
            ),
        )
        additional_lag_s = i3.number_input(
            "Additional process lag after trip initiation (s)",
            min_value=0.0,
            value=0.0,
        )
        manual_irt_s = i1.number_input(
            "Manual total IRT override (s)",
            min_value=0.0,
            value=10.0,
            disabled=irt_method != "Manual IRT",
        )

    with srs_tab:
        s1, s2 = st.columns(2)
        interlock_number = s1.text_input("Interlock number", "I-1001")
        interlock_description = s2.text_input(
            "Interlock description", "High-high level shutdown"
        )
        ipl_credited = s1.selectbox(
            "IPL credited in HAZOP/LOPA", ["Yes", "No"]
        )
        process_safe_state = s1.text_area(
            "Process safe state",
            "Close inlet shutdown valve and stop inflow to the vessel.",
        )
        success_criteria = s2.text_area(
            "Success criteria",
            "Inlet flow is reduced to the verified leakage rate before the hazard endpoint is reached.",
        )

    submitted = st.form_submit_button(
        "Calculate PST / TTC", type="primary", use_container_width=True
    )

if not submitted:
    st.info("Complete the input tabs and select **Calculate PST / TTC**.")
    st.stop()

# =============================================================================
# Convert all input to SI
# =============================================================================
diameter_m = length_to_m(diameter_u, unit_system)
straight_length_m = length_to_m(straight_length_u, unit_system)
cone_height_m = length_to_m(cone_height_u, unit_system)
geometry = Geometry(
    equipment_type=equipment_type,
    diameter_m=diameter_m,
    straight_length_m=straight_length_m,
    head_type=head_type if heads_applicable else "Flat",
    number_of_heads=number_of_heads if heads_applicable else 0,
    cone_height_m=cone_height_m,
    bundle_displacement_m3=bundle_displacement_m3,
    subtract_bundle=subtract_bundle,
)
level_span_m = geometry.maximum_level_m
levels_m = {
    "Current": length_to_m(current_u, unit_system),
    "LLLL": length_to_m(llll_u, unit_system),
    "LL": length_to_m(ll_u, unit_system),
    "High Alarm": length_to_m(high_alarm_u, unit_system),
    "HHLL": length_to_m(hhll_u, unit_system),
    "High Hazard": length_to_m(high_hazard_u, unit_system),
    "Low Hazard": length_to_m(low_hazard_u, unit_system),
}
transmitter_span_m = length_to_m(transmitter_span_u, unit_system)
hhll_tolerance = (
    length_to_m(hhll_tolerance_u, unit_system)
    if tolerance_basis == "Absolute level unit"
    else hhll_tolerance_u
)
llll_tolerance = (
    length_to_m(llll_tolerance_u, unit_system)
    if tolerance_basis == "Absolute level unit"
    else llll_tolerance_u
)

effective_hhll_m, hhll_shift_m, hhll_clamped = apply_setpoint_tolerance(
    levels_m["HHLL"], "High Level", hhll_tolerance, tolerance_basis,
    transmitter_span_m, level_span_m
)
effective_llll_m, llll_shift_m, llll_clamped = apply_setpoint_tolerance(
    levels_m["LLLL"], "Low Level", llll_tolerance, tolerance_basis,
    transmitter_span_m, level_span_m
)

normal_inflow_m3_h = flow_to_m3_h(normal_inflow_u, flow_unit, density)
normal_outflow_m3_h = flow_to_m3_h(normal_outflow_u, flow_unit, density)
worst_inflow_m3_h = flow_to_m3_h(worst_inflow_u, flow_unit, density)
worst_outflow_m3_h = flow_to_m3_h(worst_outflow_u, flow_unit, density)
if pst_flow_basis == "Worst-case rates":
    pst_inflow_m3_h = worst_inflow_m3_h
    pst_outflow_m3_h = worst_outflow_m3_h
else:
    pst_inflow_m3_h = normal_inflow_m3_h
    pst_outflow_m3_h = normal_outflow_m3_h

normal_filling_m3_h = normal_inflow_m3_h - normal_outflow_m3_h
normal_draining_m3_h = normal_outflow_m3_h - normal_inflow_m3_h
pst_filling_m3_h = pst_inflow_m3_h - pst_outflow_m3_h
pst_draining_m3_h = pst_outflow_m3_h - pst_inflow_m3_h

if (
    final_element_type == "Shutdown valve"
    and valve_time_basis == "1 second per inch"
):
    final_element_response_s = valve_size_in
else:
    final_element_response_s = manual_final_element_s
irt_s = calculate_irt(
    sensor_response_s,
    logic_solver_s,
    final_element_response_s,
    additional_lag_s,
    manual_irt_s if irt_method == "Manual IRT" else None,
)

# =============================================================================
# Validation before calculation
# =============================================================================
errors: List[str] = []
warnings: List[str] = []

if diameter_m <= EPS or straight_length_m <= EPS:
    errors.append("Diameter and straight length must be positive.")
for name, level_m in levels_m.items():
    if level_m < 0.0 or level_m > level_span_m:
        errors.append(
            f"{name} is outside the valid 0 to "
            f"{level_for_display(level_span_m, unit_system):.3f} {level_unit} span."
        )
if not (
    levels_m["LLLL"] <= levels_m["LL"]
    <= levels_m["High Alarm"] <= levels_m["HHLL"]
):
    warnings.append("Expected setpoint order is LLLL ≤ LL ≤ H/HLL ≤ HHLL.")
if hhll_tolerance <= 0.0 or llll_tolerance <= 0.0:
    warnings.append(
        "A setpoint tolerance is zero or missing. Use the approved project/CSE value."
    )
if hhll_clamped or llll_clamped:
    warnings.append(
        "A tolerance-adjusted setpoint exceeded the level span and was clamped."
    )
if (
    geometry.is_kettle
    and geometry.subtract_bundle
    and geometry.bundle_displacement_m3 > total_internal_volume(geometry, gross=True)
):
    errors.append("Tube-bundle displacement exceeds gross internal volume.")
if ipl_credited == "No":
    warnings.append(
        "IPL Credited is No. This PST/TTC result is documentation only and must not imply IPL credit."
    )

if scenario == "High Level":
    effective_trip_m = effective_hhll_m
    nominal_trip_m = levels_m["HHLL"]
    hazard_m = levels_m["High Hazard"]
    time_to_trip_rate_m3_h = normal_filling_m3_h
    pst_rate_m3_h = pst_filling_m3_h
    if hazard_m <= effective_trip_m:
        errors.append("High-level hazard endpoint must be above effective HHLL.")
    if levels_m["Current"] >= effective_trip_m:
        warnings.append("Current level is already at or above effective HHLL.")
    if levels_m["Current"] >= hazard_m:
        warnings.append("Current level is already at or beyond the high hazard endpoint.")
    if time_to_trip_rate_m3_h <= EPS:
        warnings.append("Normal net flow does not support prospective filling.")
    if pst_rate_m3_h <= EPS:
        errors.append("Selected PST flow basis has no positive net filling rate.")
else:
    effective_trip_m = effective_llll_m
    nominal_trip_m = levels_m["LLLL"]
    hazard_m = levels_m["Low Hazard"]
    time_to_trip_rate_m3_h = normal_draining_m3_h
    pst_rate_m3_h = pst_draining_m3_h
    if hazard_m >= effective_trip_m:
        errors.append("Low-level hazard endpoint must be below effective LLLL.")
    if levels_m["Current"] <= effective_trip_m:
        warnings.append("Current level is already at or below effective LLLL.")
    if levels_m["Current"] <= hazard_m:
        warnings.append("Current level is already at or beyond the low hazard endpoint.")
    if time_to_trip_rate_m3_h <= EPS:
        warnings.append("Normal net flow does not support prospective draining.")
    if pst_rate_m3_h <= EPS:
        errors.append("Selected PST flow basis has no positive net draining rate.")

for message in warnings:
    st.warning(message)
for message in errors:
    st.error(message)
if errors:
    st.stop()

# =============================================================================
# Inventory and three independent time calculations
# =============================================================================
current_volume_m3 = vessel_volume_at_level(geometry, levels_m["Current"])
effective_trip_volume_m3 = vessel_volume_at_level(geometry, effective_trip_m)
hazard_volume_m3 = vessel_volume_at_level(geometry, hazard_m)

# Current to effective trip only. This is NOT PST.
time_to_trip_s = calculate_time_to_trip(
    current_volume_m3,
    effective_trip_volume_m3,
    time_to_trip_rate_m3_h,
    scenario,
)
if (
    (scenario == "High Level" and levels_m["Current"] >= effective_trip_m)
    or (scenario == "Low Level" and levels_m["Current"] <= effective_trip_m)
):
    time_to_trip_s = 0.0

# Effective trip to hazard endpoint only. Current level is not an argument.
pst_s = calculate_pst_time_to_consequence(
    effective_trip_volume_m3,
    hazard_volume_m3,
    pst_rate_m3_h,
    scenario,
)

# Current to hazard, reported separately and never labelled PST.
total_time_to_hazard_s = calculate_total_time_to_hazard(
    current_volume_m3,
    hazard_volume_m3,
    time_to_trip_rate_m3_h,
    scenario,
)
if (
    (scenario == "High Level" and levels_m["Current"] >= hazard_m)
    or (scenario == "Low Level" and levels_m["Current"] <= hazard_m)
):
    total_time_to_hazard_s = 0.0

assessment = evaluate_pst_vs_irt(pst_s, irt_s)
residual_inventory_change_m3 = pst_rate_m3_h * irt_s / 3600.0
hazard_reached_before_completion = pst_s is not None and irt_s >= pst_s
remaining_inventory_margin_m3 = (
    abs(hazard_volume_m3 - effective_trip_volume_m3)
    - residual_inventory_change_m3
)

# =============================================================================
# Results
# =============================================================================
st.header("Results")
metric_row_1 = st.columns(4)
metric_row_1[0].metric(
    "Total internal volume", f"{total_internal_volume(geometry):,.3f} m³"
)
metric_row_1[1].metric("Time to Trip", seconds_text(time_to_trip_s))
metric_row_1[2].metric("PST / TTC", seconds_text(pst_s))
metric_row_1[3].metric(
    "Total Time to Hazard", seconds_text(total_time_to_hazard_s)
)
metric_row_2 = st.columns(4)
metric_row_2[0].metric("IRT", f"{irt_s:,.1f} s")
ratio = assessment["pst_irt_ratio"]
ratio_text = "∞" if ratio == math.inf else (f"{ratio:.2f}" if ratio is not None else "N/A")
metric_row_2[1].metric("PST / IRT ratio", ratio_text)
metric_row_2[2].metric(
    "IRT / PST",
    f"{assessment['irt_pst_percent']:.1f}%"
    if assessment["irt_pst_percent"] is not None
    else "N/A",
)
metric_row_2[3].metric("Adequacy", str(assessment["status"]))

st.markdown("#### Time-definition audit trail")
time_basis = pd.DataFrame(
    [
        {
            "Reported time": "Time to Trip",
            "Start": "Current liquid level",
            "End": "Effective conservative trip setpoint",
            "Flow basis": "Normal net directional flow",
            "Result (s)": time_to_trip_s,
        },
        {
            "Reported time": "PST / TTC",
            "Start": "Effective conservative trip setpoint",
            "End": "Hazardous event endpoint",
            "Flow basis": pst_flow_basis,
            "Result (s)": pst_s,
        },
        {
            "Reported time": "Total Time to Hazard",
            "Start": "Current liquid level",
            "End": "Hazardous event endpoint",
            "Flow basis": "Normal net directional flow",
            "Result (s)": total_time_to_hazard_s,
        },
    ]
)
st.dataframe(time_basis, use_container_width=True, hide_index=True)

if assessment["status"] == "PREFERRED":
    st.success("PREFERRED: IRT ≤ PST/2 and IRT < PST.")
elif assessment["status"] == "WARNING":
    st.warning("WARNING: PST/2 < IRT < PST. Basic check passes, preferred target does not.")
elif assessment["status"] == "FAIL":
    st.error("FAIL: IRT ≥ PST. Hazard endpoint is reached before or at action completion.")
else:
    st.warning("PST versus IRT could not be evaluated.")

residual_name = (
    "Residual accumulation during IRT"
    if scenario == "High Level"
    else "Residual depletion during IRT"
)
st.write(f"**{residual_name}:** {residual_inventory_change_m3:,.4f} m³")
st.write(
    "**Remaining inventory margin at IRT completion:** "
    f"{remaining_inventory_margin_m3:,.4f} m³"
)
st.write(
    "**Hazard endpoint reached before safety action completes:** "
    f"{'Yes' if hazard_reached_before_completion else 'No'}"
)

setpoint_table = pd.DataFrame(
    [
        {
            "Point": "Current level",
            f"Level ({level_unit})": level_for_display(levels_m["Current"], unit_system),
            "Net volume (m3)": current_volume_m3,
        },
        {
            "Point": "Nominal trip setpoint",
            f"Level ({level_unit})": level_for_display(nominal_trip_m, unit_system),
            "Net volume (m3)": vessel_volume_at_level(geometry, nominal_trip_m),
        },
        {
            "Point": "Effective conservative trip setpoint",
            f"Level ({level_unit})": level_for_display(effective_trip_m, unit_system),
            "Net volume (m3)": effective_trip_volume_m3,
        },
        {
            "Point": "Hazard endpoint",
            f"Level ({level_unit})": level_for_display(hazard_m, unit_system),
            "Net volume (m3)": hazard_volume_m3,
        },
    ]
)
st.subheader("Setpoint and inventory basis")
st.dataframe(setpoint_table, use_container_width=True, hide_index=True)

if geometry.is_kettle:
    kettle_rows = []
    for point, level_m in {
        "Current": levels_m["Current"],
        "Effective trip": effective_trip_m,
        "Hazard endpoint": hazard_m,
    }.items():
        gross_volume, net_volume = kettle_liquid_volume(geometry, level_m)
        kettle_rows.append(
            {
                "Point": point,
                "Gross liquid volume (m3)": gross_volume,
                "Net liquid volume (m3)": net_volume,
                "Allocated displacement (m3)": gross_volume - net_volume,
            }
        )
    st.subheader("Kettle gross and net inventory")
    st.dataframe(pd.DataFrame(kettle_rows), use_container_width=True, hide_index=True)

figure = go.Figure()
plot_items = [
    ("Current", levels_m["Current"], "#1f77b4", "solid"),
    ("Effective trip", effective_trip_m, "#ff7f0e", "dash"),
    ("Hazard endpoint", hazard_m, "#d62728", "dot"),
]
for name, level_m, color, dash in plot_items:
    displayed_level = level_for_display(level_m, unit_system)
    figure.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[displayed_level, displayed_level],
            mode="lines",
            name=name,
            line={"color": color, "width": 4, "dash": dash},
        )
    )
figure.update_layout(
    title="Level reference plot",
    height=350,
    xaxis={"visible": False, "range": [0.0, 1.0]},
    yaxis={
        "title": f"Level ({level_unit})",
        "range": [0.0, level_for_display(level_span_m, unit_system)],
    },
    margin={"l": 20, "r": 20, "t": 50, "b": 20},
)
st.plotly_chart(figure, use_container_width=True)

# =============================================================================
# SRS / Control Systems summary
# =============================================================================
if tolerance_basis == "Absolute level unit":
    selected_tolerance_text = (
        f"{hhll_tolerance_u:g} {level_unit}"
        if scenario == "High Level"
        else f"{llll_tolerance_u:g} {level_unit}"
    )
else:
    selected_tolerance_text = (
        f"{hhll_tolerance_u:g}%"
        if scenario == "High Level"
        else f"{llll_tolerance_u:g}%"
    )

limitations_text = (
    "Preliminary steady-net-flow screening. Horizontal head partial volume "
    "uses shell fill fraction. Torispherical head and kettle bundle displacement "
    "are approximations. The model excludes flashing, foaming, entrainment, "
    "level swell, gas compression, changing flow with level, controller action, "
    "relief-system interaction, and dynamic process response. " + user_limitations
)

summary = build_pst_summary(
    {
        "Interlock number": interlock_number,
        "Interlock description": interlock_description,
        "IPL credited in HAZOP/LOPA": ipl_credited,
        "Equipment tag/name": equipment_tag,
        "Scenario type": scenario,
        f"Current level ({level_unit})": level_for_display(levels_m["Current"], unit_system),
        f"Nominal trip setpoint ({level_unit})": level_for_display(nominal_trip_m, unit_system),
        "Setpoint tolerance": selected_tolerance_text,
        "Tolerance basis": tolerance_basis,
        f"Setpoint tolerance shift ({level_unit})": level_for_display(
            hhll_shift_m if scenario == "High Level" else llll_shift_m,
            unit_system,
        ),
        f"Effective conservative trip setpoint ({level_unit})": level_for_display(
            effective_trip_m, unit_system
        ),
        f"Hazard endpoint level ({level_unit})": level_for_display(hazard_m, unit_system),
        "Current volume (m3)": current_volume_m3,
        "Effective trip volume (m3)": effective_trip_volume_m3,
        "Hazard endpoint volume (m3)": hazard_volume_m3,
        "Normal inflow (m3/h)": normal_inflow_m3_h,
        "Normal outflow (m3/h)": normal_outflow_m3_h,
        "PST/TTC flow basis": pst_flow_basis,
        "PST basis inflow (m3/h)": pst_inflow_m3_h,
        "PST basis outflow (m3/h)": pst_outflow_m3_h,
        "PST directional net flow (m3/h)": pst_rate_m3_h,
        "Time to Trip (s)": time_to_trip_s,
        "PST / TTC (s)": pst_s,
        "Total Time to Hazard (s)": total_time_to_hazard_s,
        "IRT method": irt_method,
        "Sensor response (s)": sensor_response_s,
        "Logic solver time (s)": logic_solver_s,
        "Final element response (s)": final_element_response_s,
        "Additional lag (s)": additional_lag_s,
        "IRT (s)": irt_s,
        "PST/IRT ratio": assessment["pst_irt_ratio"],
        "IRT/PST (%)": assessment["irt_pst_percent"],
        "PST > IRT check": "PASS" if assessment["pst_gt_irt"] else "FAIL",
        "IRT <= PST/2 preferred check": "PASS" if assessment["preferred"] else "NOT MET",
        "Overall assessment": assessment["status"],
        "Residual inventory change during IRT (m3)": residual_inventory_change_m3,
        "Hazard reached before action completion": "Yes" if hazard_reached_before_completion else "No",
        "Initiating variable": initiating_variable,
        "Process safe state": process_safe_state,
        "Success criteria": success_criteria,
        "Applicable operating mode": operating_mode,
        "Calculation basis": calculation_basis,
        "Data source": data_source,
        "Assumptions and limitations": limitations_text,
    }
)

st.subheader("PST Summary for Control Systems / SRS Input")
st.dataframe(summary.T.rename(columns={0: "Value"}), use_container_width=True)
csv_name = safe_filename(f"{equipment_tag}_{interlock_number}_pst_summary.csv")
st.download_button(
    "Download PST summary CSV",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name=csv_name,
    mime="text/csv",
    use_container_width=True,
)

with st.expander("PST Methodology", expanded=True):
    st.markdown(
        """
- **Time to Trip** is calculated from current level to the effective conservative trip setpoint.
- **PST / TTC** is calculated only from the effective conservative trip setpoint to the hazardous event endpoint. Current level is not an input to the PST function.
- **Total Time to Hazard** is calculated independently from current level to the hazardous endpoint.
- High-level calculations use positive net filling rate. Low-level calculations use positive net draining rate.
- IRT is independent of PST and consists of sensor response, logic-solver time, final-element response, and additional process lag, unless a verified manual total is entered.
- Basic adequacy requires **IRT < PST**. The preferred preliminary target is **IRT ≤ PST/2**.
"""
    )

with st.expander("Assumptions and Limitations", expanded=True):
    st.markdown(
        f"""
- {limitations_text}
- Validate geometry against approved GA drawings, equipment data sheets, and vendor data.
- Validate flow scenarios against approved H&MB data, valve/pump/compressor cases, and applicable failure modes.
- Validate horizontal-head, torispherical-head, and kettle bundle approximations using approved software or numerical integration where material to the decision.
- Confirm the setpoint-tolerance sign convention against project and Control Systems practice. This app implements the explicitly requested convention: high trip shifted lower and low trip shifted higher.
- Use dynamic simulation where pressure, vapor generation, flashing, control action, changing flow, heat input, or two-phase behavior materially affects TTC.
"""
    )
