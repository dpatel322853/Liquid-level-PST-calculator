"""Vessel Inventory, PST/TTC, Operating Envelope, and PDF Report.
Preliminary engineering screening only.
"""
from __future__ import annotations
import math, re
from html import escape
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")  # Required for headless Streamlit Cloud/container deployment.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

EPS=1e-12
HEADS=["Flat","2:1 ellipsoidal","Hemispherical","Torispherical (approx.)","Conical (approx.)"]
EQUIPMENT=["Vertical cylindrical vessel","Horizontal cylindrical vessel","Horizontal vessel with heads","Vertical vessel with heads","Kettle exchanger / reboiler"]

@dataclass(frozen=True)
class Geometry:
    equipment_type:str; diameter_m:float; straight_length_m:float
    head_type:str="Flat"; number_of_heads:int=0; cone_height_m:float=0.0
    bundle_displacement_m3:float=0.0; subtract_bundle:bool=False
    @property
    def vertical(self): return self.equipment_type.startswith("Vertical")
    @property
    def kettle(self): return self.equipment_type.startswith("Kettle")
    @property
    def max_level_m(self):
        return self.straight_length_m+self.number_of_heads*head_depth(self.diameter_m,self.head_type,self.cone_height_m) if self.vertical else self.diameter_m

def clamp(x,a,b): return max(a,min(x,b))
def segment_area(d,h):
    r=d/2; h=clamp(h,0,d)
    if h<=0:return 0.0
    if h>=d:return math.pi*r*r
    return r*r*math.acos((r-h)/r)-(r-h)*math.sqrt(max(0,2*r*h-h*h))
def head_depth(d,t,ch=0):
    return {"Flat":0,"2:1 ellipsoidal":d/4,"Hemispherical":d/2,"Torispherical (approx.)":0.1935*d}.get(t,ch if ch>0 else d/4)
def head_volume(d,t,ch=0):
    r=d/2
    if t=="Flat":return 0
    if t=="2:1 ellipsoidal":return 2/3*math.pi*r*r*(d/4)
    if t=="Hemispherical":return 2/3*math.pi*r**3
    if t=="Torispherical (approx.)":return 0.084*d**3
    return math.pi*r*r*head_depth(d,t,ch)/3
def bottom_head_partial(d,t,h,ch=0):
    dep=head_depth(d,t,ch); full=head_volume(d,t,ch)
    if dep<=EPS:return 0
    x=clamp(h/dep,0,1)
    f=x**3 if t=="Conical (approx.)" else (1.5*x*x-0.5*x**3 if t in ["2:1 ellipsoidal","Hemispherical"] else 3*x*x-2*x**3)
    return full*f
def vessel_volume_at_level(g:Geometry,level:float,gross=False):
    d,L=g.diameter_m,g.straight_length_m; level=clamp(level,0,g.max_level_m); hv=head_volume(d,g.head_type,g.cone_height_m)
    if g.vertical:
        hd=head_depth(d,g.head_type,g.cone_height_m); bottom=g.number_of_heads>=1; top=g.number_of_heads>=2
        if bottom and level<hd:v=bottom_head_partial(d,g.head_type,level,g.cone_height_m)
        else:
            v=hv if bottom else 0; origin=hd if bottom else 0; sh=clamp(level-origin,0,L); v+=math.pi*d*d/4*sh
            if top and level>origin+L:
                p=clamp(level-origin-L,0,hd); v+=hv-bottom_head_partial(d,g.head_type,hd-p,g.cone_height_m)
    else:
        v=segment_area(d,level)*L
        if g.number_of_heads:
            v+=g.number_of_heads*hv*segment_area(d,level)/(math.pi*d*d/4)
    if g.kettle and g.subtract_bundle and not gross:v-=g.bundle_displacement_m3*clamp(level/d,0,1)
    return max(0,v)
def apply_tolerance(nominal,scenario,value,basis,tx_span,vessel_span):
    shift=value if basis=="Absolute level unit" else value/100*(tx_span if basis.startswith("Percent of transmitter") else vessel_span)
    raw=nominal-shift if scenario=="High Level" else nominal+shift
    return clamp(raw,0,vessel_span),shift,not math.isclose(raw,clamp(raw,0,vessel_span),abs_tol=EPS)
def time_s(dv,q): return None if dv<0 or q<=EPS else dv/q*3600
def calculate_irt(sensor,logic,final,lag,manual=None): return manual if manual is not None else sensor+logic+final+lag
def evaluate(pst,irt):
    if pst is None or pst<=0:return dict(status="NOT AVAILABLE",ratio=None,pct=None,preferred=False,passed=False,color="#6b7280")
    if irt>=pst:status,color="FAIL","#dc2626"
    elif irt<=pst/2:status,color="PREFERRED","#16a34a"
    else:status,color="WARNING","#d97706"
    return dict(status=status,ratio=(math.inf if irt==0 else pst/irt),pct=100*irt/pst,preferred=irt<=pst/2,passed=irt<pst,color=color)
def display_level(x,unit): return x*1000 if unit=="mm" else x
def to_m(x,unit): return x/1000 if unit=="mm" else x
def safe_name(s): return re.sub(r"[^A-Za-z0-9_.-]+","_",s).strip("_") or "report"
def fmt_number(value, decimals=2):
    if value is None or not math.isfinite(value): return "Not available"
    return f"{value:,.{decimals}f}"
def pdf_text(value):
    return escape(str(value), quote=False).replace("\n", "<br/>")

def envelope_bands(levels):
    """Generic vertical alarm/trip diagram. Boundaries come from the calculation inputs."""
    return [(0,levels["LDL"],"Unsafe Zone","#f3f4f6"),(levels["LDL"],levels["LL"],"Lower Risk Zone","#ef4444"),(levels["LL"],levels["L"],"Lower Alarming Zone","#fed7aa"),(levels["L"],levels["H"],"Target Operating Range","#86efac"),(levels["H"],levels["HH"],"Upper Alarming Zone","#fed7aa"),(levels["HH"],levels["UDL"],"Upper Risk Zone","#ef4444"),(levels["UDL"],levels["SPAN"],"Unsafe Zone","#f3f4f6")]
def check_text(a):
    if not a or a["ratio"] is None:return "PST/IRT check not available","#6b7280"
    if a["status"]=="FAIL":return "PST <= IRT",a["color"]
    return "PST > IRT",a["color"]
def create_operating_envelope_plot(levels,volumes,unit,tag,current,hazard,effective,irt,assessment,ttt,pst,total,scenario):
    fig=go.Figure(); x0,x1=0.15,0.60
    for lo,hi,name,color in envelope_bands(levels):
        if hi<=lo: continue
        fig.add_shape(type="rect",x0=x0,x1=x1,y0=lo,y1=hi,fillcolor=color,line=dict(color="white",width=1),layer="below")
        fig.add_trace(go.Scatter(x=[(x0+x1)/2],y=[(lo+hi)/2],mode="text",text=[name],textfont=dict(size=11),hovertemplate=f"{name}<br>{lo:.3g} to {hi:.3g} {unit}<extra></extra>",showlegend=False))
    pts=[("Upper Design Limit","UDL","black","dash"),("HH trip","HH","#b91c1c","dash"),("H alarm","H","#c2410c","dash"),("Normal setpoint","NORMAL","#15803d","dot"),("L alarm","L","#c2410c","dash"),("LL trip","LL","#b91c1c","dash"),("Lower Design Limit","LDL","black","dash")]
    for label,key,color,dash in pts:
        y=levels[key]; v=volumes.get(key); v_text=f"{v:.3f} m3" if v is not None else "Not available"
        custom=(f"{label}<br>Level: {y:.3g} {unit}<br>Volume: {v_text}"
                f"<br>Difference from current: {y-current:+.3g} {unit}"
                f"<br>Time to Trip: {fmt_number(ttt)} s<br>PST/TTC: {fmt_number(pst)} s"
                f"<br>IRT: {irt:.2f} s<br>PST/IRT ratio: {fmt_number(assessment.get('ratio'),3)}"
                f"<br>Status: {assessment['status']}")
        fig.add_trace(go.Scatter(x=[x0,x1],y=[y,y],mode="lines",line=dict(color=color,dash=dash,width=2),name=label,hovertemplate=custom+"<extra></extra>"))
        fig.add_annotation(x=0.64,y=y,text=f"{label}: {y:.3g} {unit}",showarrow=False,xanchor="left",font=dict(size=10,color=color))
    fig.add_trace(go.Scatter(x=[x0,x1],y=[current,current],mode="lines",line=dict(color="#2563eb",width=4),name="Current level",hovertemplate=f"Current level: {current:.3g} {unit}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[x0,x1],y=[hazard,hazard],mode="lines",line=dict(color="#7e22ce",width=3,dash="dot"),name="Selected hazard endpoint",hovertemplate=f"Selected hazard endpoint: {hazard:.3g} {unit}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[x0,x1],y=[effective,effective],mode="lines",line=dict(color="#7f1d1d",width=3,dash="dashdot"),name="Effective conservative trip",hovertemplate=f"Effective trip: {effective:.3g} {unit}<extra></extra>"))
    selected_text,selected_color=check_text(assessment); unavailable="PST/IRT check not available"
    upper_text,upper_color=(selected_text,selected_color) if scenario=="High Level" else (unavailable,"#6b7280")
    lower_text,lower_color=(selected_text,selected_color) if scenario=="Low Level" else (unavailable,"#6b7280")
    fig.add_annotation(x=0.90,y=(levels["HH"]+levels["UDL"])/2,text=upper_text,showarrow=False,textangle=-90,font=dict(color=upper_color,size=10),bordercolor=upper_color,borderwidth=1,bgcolor="white")
    fig.add_annotation(x=0.90,y=(levels["LDL"]+levels["LL"])/2,text=lower_text,showarrow=False,textangle=-90,font=dict(color=lower_color,size=10),bordercolor=lower_color,borderwidth=1,bgcolor="white")
    fig.add_annotation(x=0.82,y=(levels["H"]+levels["HH"])/2,text="Response time + Margin / TTE",showarrow=False,textangle=-90,font=dict(size=9))
    fig.add_annotation(x=0.82,y=(levels["LL"]+levels["L"])/2,text="Response time + Margin / TTE",showarrow=False,textangle=-90,font=dict(size=9))
    fig.add_annotation(x=0.77,y=levels["NORMAL"],text="System swing range + Margin",showarrow=False,font=dict(size=9,color="#166534"))
    fig.add_annotation(x=x0,y=levels["SPAN"]*1.03,text="IN",showarrow=True,ax=-45,ay=0,arrowhead=2)
    fig.add_annotation(x=x0,y=0,text="OUT",showarrow=True,ax=-45,ay=0,arrowhead=2)
    fig.update_layout(title=f"Operating Envelope and PST Visualization - {tag}",xaxis=dict(visible=False,range=[0,1.15]),yaxis=dict(title=f"Level ({unit})",range=[0,levels["SPAN"]*1.08]),height=720,legend=dict(orientation="h",y=-.08),margin=dict(l=50,r=40,t=70,b=80),hovermode="closest")
    return fig

def create_matplotlib_operating_envelope_plot(levels,unit,tag,current,hazard,effective,assessment,scenario):
    fig,ax=plt.subplots(figsize=(7.2,8.4)); x0,w=.12,.43
    for lo,hi,name,color in envelope_bands(levels):
        if hi<=lo:continue
        ax.add_patch(Rectangle((x0,lo),w,hi-lo,facecolor=color,edgecolor="white")); ax.text(x0+w/2,(lo+hi)/2,name,ha="center",va="center",fontsize=8)
    for label,key,color in [("Upper Design Limit","UDL","black"),("HH trip","HH","darkred"),("H alarm","H","darkorange"),("Normal","NORMAL","darkgreen"),("L alarm","L","darkorange"),("LL trip","LL","darkred"),("Lower Design Limit","LDL","black")]:
        y=levels[key]; ax.hlines(y,x0,x0+w,color=color,linestyle="--",linewidth=1.3); ax.text(.58,y,f"{label}: {y:.3g} {unit}",va="center",fontsize=8,color=color)
    ax.hlines(current,x0,x0+w,color="royalblue",linewidth=3,label="Current")
    if hazard is not None:ax.hlines(hazard,x0,x0+w,color="purple",linestyle=":",linewidth=2,label="Hazard endpoint")
    if effective is not None:ax.hlines(effective,x0,x0+w,color="maroon",linestyle="-.",linewidth=2,label="Effective trip")
    text,color=check_text(assessment); unavailable="PST/IRT check not available"
    ut,uc=(text,color) if scenario=="High Level" else (unavailable,"#6b7280")
    lt,lc=(text,color) if scenario=="Low Level" else (unavailable,"#6b7280")
    ax.text(.82,(levels["HH"]+levels["UDL"])/2,ut,rotation=90,ha="center",va="center",color=uc,fontsize=8,bbox=dict(facecolor="white",edgecolor=uc))
    ax.text(.90,(levels["LDL"]+levels["LL"])/2,lt,rotation=90,ha="center",va="center",color=lc,fontsize=8,bbox=dict(facecolor="white",edgecolor=lc))
    ax.set_xlim(0,1); ax.set_ylim(0,levels["SPAN"]*1.08); ax.set_xticks([]); ax.set_ylabel(f"Level ({unit})"); ax.set_title(f"Operating Envelope and PST Visualization\n{tag}"); ax.grid(axis="y",alpha=.12); ax.legend(loc="lower center",bbox_to_anchor=(.5,-.08),ncol=3,fontsize=7); fig.tight_layout(); return fig

def image_bytes(fig):
    b=BytesIO(); fig.savefig(b,format="png",dpi=180,bbox_inches="tight"); plt.close(fig); b.seek(0); return b

def build_pdf_report(report,plot_png,warnings):
    """Build an in-memory engineering screening report using ReportLab."""
    out=BytesIO(); styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name="Title2",parent=styles["Title"],alignment=TA_CENTER,textColor=colors.HexColor("#17365D"),fontSize=18,leading=22)); styles.add(ParagraphStyle(name="Small",parent=styles["BodyText"],fontSize=8,leading=10))
    def footer(canvas,doc):
        canvas.saveState(); canvas.setFont("Helvetica",8); canvas.drawString(18*mm,10*mm,"Preliminary calculation. Engineering verification required."); canvas.drawRightString(192*mm,10*mm,f"Page {doc.page}"); canvas.restoreState()
    doc=SimpleDocTemplate(out,pagesize=A4,rightMargin=16*mm,leftMargin=16*mm,topMargin=15*mm,bottomMargin=17*mm)
    story=[Paragraph("Vessel Liquid Inventory and Process Safety Time Calculation Report",styles["Title2"]),Spacer(1,5*mm)]
    story.append(Table([[Paragraph(pdf_text(k),styles["Small"]),Paragraph(pdf_text(v),styles["Small"])] for k,v in [["Generated",report["Generated"]],["Equipment",report["Equipment tag/name"]],["Interlock",report["Interlock number"]],["Description",report["Interlock description"]],["Scenario",report["Scenario"]],["Prepared by",report["Prepared by"]]]],colWidths=[42*mm,130*mm],style=[("BACKGROUND",(0,0),(0,-1),colors.HexColor("#D9EAF7")),("GRID",(0,0),(-1,-1),.4,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP"),("FONTSIZE",(0,0),(-1,-1),9),("LEFTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)])); story+= [Spacer(1,4*mm),Paragraph("Safety Disclaimer",styles["Heading2"]),Paragraph(pdf_text(report["Disclaimer"]),styles["BodyText"])]
    for title,key in [("Input Summary","Inputs"),("Calculation Results","Results"),("SRS-Style Summary","SRS")]:
        story += [Spacer(1,3*mm),Paragraph(title,styles["Heading2"])]
        rows=[[Paragraph("Parameter",styles["Small"]),Paragraph("Value",styles["Small"])]]+[[Paragraph(pdf_text(k),styles["Small"]),Paragraph(pdf_text(v),styles["Small"])] for k,v in report[key].items()]
        t=Table(rows,colWidths=[65*mm,107*mm],repeatRows=1); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17365D")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.3,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F5F7FA")]),("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)])); story.append(t)
    story += [PageBreak(),Paragraph("Operating Envelope and PST Visualization",styles["Heading2"]),Image(plot_png,width=170*mm,height=198*mm),Spacer(1,3*mm),Paragraph("Warnings and Validation Messages",styles["Heading2"])]
    story.append(Paragraph("<br/>".join([f"- {pdf_text(w)}" for w in warnings]) if warnings else "No active warnings.",styles["BodyText"]))
    story += [Spacer(1,3*mm),Paragraph("PST Methodology",styles["Heading2"]),Paragraph("Time to Trip is current level to effective conservative trip. PST/TTC is effective conservative trip to the hazardous endpoint if the safety function does not act. Total Time to Hazard is current level to hazardous endpoint. IRT is calculated separately and checked against PST and the preferred PST/2 target.",styles["BodyText"])]
    doc.build(story,onFirstPage=footer,onLaterPages=footer); out.seek(0); return out.getvalue()

st.set_page_config(page_title="Vessel Inventory & PST/TTC",page_icon="🛡️",layout="wide")
st.title("Vessel Inventory and PST / TTC Screening Calculator")

with st.sidebar:
    unit_system=st.selectbox("Unit system",["SI (m, m3, m3/h)","Engineering (mm, m3, m3/h)"]); unit="mm" if unit_system.startswith("Engineering") else "m"; scenario=st.radio("Scenario",["High Level","Low Level"])
with st.form("inputs"):
    tg,tl,tf,tp,ti,ts=st.tabs(["Equipment Geometry","Level Setpoints","Flow Basis","PST/TTC Basis","IRT Basis","SRS Documentation"])
    with tg:
        c=st.columns(3); tag=c[0].text_input("Equipment tag/name","V-1001"); et=c[1].selectbox("Equipment type",EQUIPMENT); D_u=c[2].number_input(f"Internal diameter ({unit})",.001,value=2.0 if unit=="m" else 2000.0); L_u=c[0].number_input(f"Straight length ({unit})",.001,value=5.0 if unit=="m" else 5000.0); heads_on="with heads" in et.lower() or et.startswith("Kettle"); ht=c[1].selectbox("Head type",HEADS,disabled=not heads_on); nh=c[2].selectbox("Number of heads",[0,1,2],index=2 if heads_on else 0,disabled=not heads_on); ch_u=c[0].number_input(f"Cone depth ({unit})",0.0,value=0.0); bundle=c[1].number_input("Bundle displacement (m3)",0.0,value=0.0,disabled=not et.startswith("Kettle")); subtract=c[2].checkbox("Subtract bundle displacement",True,disabled=not et.startswith("Kettle"))
    with tl:
        c=st.columns(3); current_u=c[0].number_input(f"Current level ({unit})",0.0,value=1.0 if unit=="m" else 1000.0); normal_u=c[1].number_input(f"Normal setpoint ({unit})",0.0,value=1.0 if unit=="m" else 1000.0); llll_u=c[2].number_input(f"LL/LLLL trip ({unit})",0.0,value=.30 if unit=="m" else 300.0); ll_u=c[0].number_input(f"L alarm ({unit})",0.0,value=.50 if unit=="m" else 500.0); h_u=c[1].number_input(f"H alarm ({unit})",0.0,value=1.50 if unit=="m" else 1500.0); hh_u=c[2].number_input(f"HH/HHLL trip ({unit})",0.0,value=1.70 if unit=="m" else 1700.0); ldl_u=c[0].number_input(f"Lower design limit ({unit})",0.0,value=.10 if unit=="m" else 100.0); udl_u=c[1].number_input(f"Upper design limit ({unit})",0.0,value=1.90 if unit=="m" else 1900.0); lowhaz_u=c[2].number_input(f"Low hazard endpoint ({unit})",0.0,value=.10 if unit=="m" else 100.0); highhaz_u=c[0].number_input(f"High hazard endpoint ({unit})",0.0,value=1.90 if unit=="m" else 1900.0); tb=c[1].selectbox("Tolerance basis",["Percent of transmitter calibrated range","Percent of vessel level span","Absolute level unit"]); tv=c[2].number_input("Selected trip tolerance (% or absolute)",0.0,value=2.0 if tb!="Absolute level unit" else (0.02 if unit=="m" else 20.0)); tx_u=c[0].number_input(f"Transmitter span ({unit})",.001,value=2.0 if unit=="m" else 2000.0)
    with tf:
        c=st.columns(3); fu=c[0].selectbox("Flow unit",["m3/h","kg/h"]); rho=c[1].number_input("Density (kg/m3)",.001,value=850.0); flow_basis=c[2].radio("PST flow basis",["Worst-case rates","Normal rates"]); ni=c[0].number_input(f"Normal inflow ({fu})",0.0,value=100.0); no=c[1].number_input(f"Normal outflow ({fu})",0.0,value=60.0); wi=c[0].number_input(f"Worst inflow ({fu})",0.0,value=120.0); wo=c[1].number_input(f"Worst outflow ({fu})",0.0,value=20.0)
    with tp:
        c=st.columns(2); mode=c[0].selectbox("Operating mode",["Startup","Normal operation","Turndown","Shutdown","Recycle","Maintenance","Bypass","Other"],index=1); basis=c[0].text_area("Calculation basis","Steady net flow and static geometry; preliminary screening."); source=c[1].text_area("Data source","Equipment datasheet, P&ID, H&MB, setpoint summary."); limitations=c[1].text_area("Assumptions and limitations","No flashing, foaming, level swell, changing flow, or dynamic interactions.")
    with ti:
        c=st.columns(3); im=c[0].radio("IRT method",["Preliminary calculation","Manual IRT"]); sensor=c[1].number_input("Sensor response (s)",0.0,value=1.0); logic=c[2].number_input("Logic solver (s)",0.0,value=1.0); finaltype=c[0].selectbox("Final element",["Shutdown valve","Control valve","Pump trip","Compressor trip","Other"]); size=c[1].number_input("Valve size (inch)",0.0,value=6.0); finalmanual=c[2].number_input("Manual final element time (s)",0.0,value=5.0); lag=c[0].number_input("Additional lag (s)",0.0,value=0.0); manualirt=c[1].number_input("Manual total IRT (s)",0.0,value=10.0)
    with ts:
        c=st.columns(2); ino=c[0].text_input("Interlock number","I-1001"); idesc=c[1].text_input("Interlock description","Level shutdown"); ipl=c[0].selectbox("IPL credited",["Yes","No"]); prepared=c[1].text_input("Prepared by",""); safe=c[0].text_area("Process safe state","Stop inflow / isolate applicable source."); success=c[1].text_area("Success criteria","Safe state achieved before hazard endpoint.")
    submit=st.form_submit_button("Calculate",type="primary",use_container_width=True)

# Invalidate a previously generated report whenever the calculation form is resubmitted.
# This prevents a stale PDF from being downloaded after inputs or setpoints change.
if submit:
    st.session_state.pop("pdf_report_bytes", None)
    st.session_state.pop("pdf_report_name", None)

D,L,ch=to_m(D_u,unit),to_m(L_u,unit),to_m(ch_u,unit); g=Geometry(et,D,L,ht if heads_on else "Flat",nh if heads_on else 0,ch,bundle,subtract); span=g.max_level_m
lm={"CURRENT":to_m(current_u,unit),"NORMAL":to_m(normal_u,unit),"LL":to_m(llll_u,unit),"L":to_m(ll_u,unit),"H":to_m(h_u,unit),"HH":to_m(hh_u,unit),"LDL":to_m(ldl_u,unit),"UDL":to_m(udl_u,unit),"LOW_HAZ":to_m(lowhaz_u,unit),"HIGH_HAZ":to_m(highhaz_u,unit),"SPAN":span}
tol=to_m(tv,unit) if tb=="Absolute level unit" else tv; tx=to_m(tx_u,unit); effective,shift,clamped=apply_tolerance(lm["HH"] if scenario=="High Level" else lm["LL"],scenario,tol,tb,tx,span)
def q(v): return v/rho if fu=="kg/h" else v
ni,no,wi,wo=map(q,[ni,no,wi,wo]); pi,po=(wi,wo) if flow_basis=="Worst-case rates" else (ni,no); normal_rate=ni-no if scenario=="High Level" else no-ni; pst_rate=pi-po if scenario=="High Level" else po-pi
final=size if finaltype=="Shutdown valve" else finalmanual; irt=calculate_irt(sensor,logic,final,lag,manualirt if im=="Manual IRT" else None)
warn=[]
if not (lm["LDL"]<lm["LL"]<lm["L"]<lm["NORMAL"]<lm["H"]<lm["HH"]<lm["UDL"]):warn.append("Standard ordering LDL < LL < L < Normal < H < HH < UDL is not satisfied. Plot uses the entered boundaries where possible.")
for k,v in lm.items():
    if k!="SPAN" and not 0<=v<=span:warn.append(f"{k} lies outside the vessel level span.")
if clamped:warn.append("Effective trip point was clamped to the vessel span.")
if normal_rate<=0:warn.append("Normal flow basis does not support Time to Trip or Total Time to Hazard for the selected scenario.")
if pst_rate<=0:warn.append("Selected PST flow basis does not support the chosen directional scenario.")
if scenario=="High Level" and lm["HIGH_HAZ"]<=effective:warn.append("High-level hazard endpoint is not above the effective HH/HHLL trip point.")
if scenario=="Low Level" and lm["LOW_HAZ"]>=effective:warn.append("Low-level hazard endpoint is not below the effective LL/LLLL trip point.")
if ipl=="No":warn.append("IPL credited is No; PST is documentation only and must not imply IPL credit.")
trip=effective; hazard=lm["HIGH_HAZ"] if scenario=="High Level" else lm["LOW_HAZ"]; cv=vessel_volume_at_level(g,lm["CURRENT"]); ev=vessel_volume_at_level(g,trip); hv=vessel_volume_at_level(g,hazard)
ttt=time_s((ev-cv) if scenario=="High Level" else (cv-ev),normal_rate); pst=time_s((hv-ev) if scenario=="High Level" else (ev-hv),pst_rate); total=time_s((hv-cv) if scenario=="High Level" else (cv-hv),normal_rate); a=evaluate(pst,irt)
if a["status"]=="FAIL":warn.append("IRT is greater than or equal to PST.")
elif a["status"]=="WARNING":warn.append("IRT is below PST but exceeds the preferred PST/2 target.")
for w in warn:st.warning(w)
vol_m={k:vessel_volume_at_level(g,v) for k,v in lm.items() if k!="SPAN"}; levels_d={k:display_level(v,unit) for k,v in lm.items()}; effective_d=display_level(effective,unit); hazard_d=display_level(hazard,unit)

st.header("Calculation Results"); c=st.columns(5); c[0].metric("Current volume",f"{cv:.3f} m3"); c[1].metric("Time to Trip",f"{ttt:.1f} s" if ttt is not None else "N/A"); c[2].metric("PST/TTC",f"{pst:.1f} s" if pst is not None else "N/A"); c[3].metric("IRT",f"{irt:.1f} s"); c[4].metric("Status",a["status"])

st.header("Operating Envelope and PST Visualization")
plot=create_operating_envelope_plot(levels_d,{k:vol_m.get(k) for k in ["LDL","LL","L","NORMAL","H","HH","UDL"]},unit,tag,levels_d["CURRENT"],hazard_d,effective_d,irt,a,ttt,pst,total,scenario); st.plotly_chart(plot,use_container_width=True)
st.caption("Generic alarm/trip operating-zone sketch for engineering communication. It is not an approved project drawing or alarm rationalization record.")

summary=pd.DataFrame([{"Equipment":tag,"Interlock":ino,"Scenario":scenario,"Current level":levels_d["CURRENT"],"Nominal trip":levels_d["HH" if scenario=="High Level" else "LL"],"Effective trip":effective_d,"Hazard endpoint":hazard_d,"Time to Trip (s)":ttt,"PST/TTC (s)":pst,"IRT (s)":irt,"PST/IRT":a["ratio"],"IRT <= PST/2":a["preferred"],"Status":a["status"],"Safe state":safe,"Success criteria":success}])
st.header("PST Summary for Control Systems / SRS Input"); st.dataframe(summary.T.rename(columns={0:"Value"}),use_container_width=True); st.download_button("Download CSV Summary",summary.to_csv(index=False).encode(),safe_name(f"{tag}_{ino}_pst_summary.csv"),"text/csv")

st.header("PDF Calculation Report")
inputs={"Equipment type":et,"Diameter":f"{D_u:g} {unit}","Straight length":f"{L_u:g} {unit}","Head type / count":f"{ht} / {nh}","Unit system":unit_system,"Density":f"{rho:g} kg/m3" if fu=="kg/h" else "Not used","Current / Normal":f"{current_u:g} / {normal_u:g} {unit}","L / LL":f"{ll_u:g} / {llll_u:g} {unit}","H / HH":f"{h_u:g} / {hh_u:g} {unit}","Lower / Upper design limit":f"{ldl_u:g} / {udl_u:g} {unit}","Low / High hazard":f"{lowhaz_u:g} / {highhaz_u:g} {unit}","Effective trip":f"{effective_d:.3g} {unit}","Tolerance":f"{tv:g} ({tb})","Normal inflow / outflow":f"{ni:.3g} / {no:.3g} m3/h","PST inflow / outflow":f"{pi:.3g} / {po:.3g} m3/h","IRT basis":f"Sensor {sensor:g} + logic {logic:g} + final {final:g} + lag {lag:g} s; method {im}"}
results={"Current volume":f"{cv:.4f} m3","Normal volume":f"{vol_m['NORMAL']:.4f} m3","L alarm volume":f"{vol_m['L']:.4f} m3","LL trip volume":f"{vol_m['LL']:.4f} m3","H alarm volume":f"{vol_m['H']:.4f} m3","HH trip volume":f"{vol_m['HH']:.4f} m3","Effective trip volume":f"{ev:.4f} m3","Hazard volume":f"{hv:.4f} m3","Time to Trip":f"{ttt:.2f} s" if ttt is not None else "Not available","PST/TTC":f"{pst:.2f} s" if pst is not None else "Not available","Total Time to Hazard":f"{total:.2f} s" if total is not None else "Not available","IRT":f"{irt:.2f} s","PST/IRT ratio":fmt_number(a["ratio"],3),"IRT <= PST/2":"Yes" if a["preferred"] else "No","Assessment":a["status"]}
srs={"IPL credited":ipl,"Process safe state":safe,"Success criteria":success,"Operating mode":mode,"Calculation basis":basis,"Data source":source,"Assumptions and limitations":limitations}
report={"Generated":datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),"Equipment tag/name":tag,"Interlock number":ino,"Interlock description":idesc,"Scenario":scenario,"Prepared by":prepared or "Not provided","Disclaimer":"This calculation is for preliminary engineering screening only and must be reviewed and approved by qualified Process, Process Safety, and Control Systems engineers before use for design, operation, HAZOP, LOPA, Process SRS, alarm rationalization, SIS design, or any safety-critical decision.","Inputs":inputs,"Results":results,"SRS":srs}
if st.button("Generate PDF Report",type="primary",use_container_width=True):
    try:
        png=image_bytes(create_matplotlib_operating_envelope_plot(levels_d,unit,tag,levels_d["CURRENT"],hazard_d,effective_d,a,scenario))
        st.session_state["pdf_report_bytes"]=build_pdf_report(report,png,warn)
        st.session_state["pdf_report_name"]=safe_name(f"vessel_pst_report_{tag}_{datetime.now():%Y%m%d}.pdf")
        st.success("PDF report generated successfully.")
    except Exception as exc:
        st.session_state.pop("pdf_report_bytes",None); st.session_state.pop("pdf_report_name",None)
        st.error(f"PDF generation failed: {exc}")
if st.session_state.get("pdf_report_bytes"):
    st.download_button("Download PDF Report",data=st.session_state["pdf_report_bytes"],file_name=st.session_state["pdf_report_name"],mime="application/pdf",use_container_width=True)

with st.expander("Assumptions and Limitations",expanded=True):
    st.write(limitations); st.write("Horizontal head partial volume, torispherical head volume, and kettle bundle displacement are screening approximations. Validate against approved project calculations, vendor data, approved software, or dynamic simulation.")
# Test: valid order; close H/HH; IRT>PST; PDF download; compare PDF and screen values.
