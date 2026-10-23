import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.units import cm
from datetime import date
import tempfile, os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="FraudLens", page_icon="🔍", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #0f1117; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 48px 40px;
        margin-bottom: 32px;
        border: 1px solid #1e3a5f;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 8px 0;
        letter-spacing: -1px;
    }
    .hero p {
        font-size: 1.1rem;
        color: #8892a4;
        margin: 0 0 4px 0;
    }
    .hero .badge {
        display: inline-block;
        background: #0f3460;
        color: #4fc3f7;
        border: 1px solid #1565c0;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        margin-top: 16px;
    }

    /* Metric cards */
    .metric-row { display: flex; gap: 16px; margin-bottom: 32px; }
    .metric-card {
        flex: 1;
        background: #1a1f2e;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        border: 1px solid #2a2f3e;
    }
    .metric-card .value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 8px 0 4px 0;
    }
    .metric-card .label { font-size: 0.85rem; color: #8892a4; }
    .total .value { color: #ffffff; }
    .high .value   { color: #ef5350; }
    .medium .value { color: #ffa726; }
    .low .value    { color: #66bb6a; }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #1e3a5f;
    }

    /* Upload area */
    .upload-area {
        background: #1a1f2e;
        border: 2px dashed #1e3a5f;
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        margin-bottom: 24px;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Hero section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔍 FraudLens</h1>
    <p>AI-Assisted Financial Statement Fraud Detection Pipeline</p>
    <p>Beneish M-Score · Z-Score Anomaly Detection · Isolation Forest ML</p>
    <span class="badge">v3.0 · Built for Forensic Accountants</span>
</div>
""", unsafe_allow_html=True)

# ── Instructions + template ───────────────────────────────────────────────────
col_a, col_b = st.columns([2, 1])
with col_a:
    st.markdown("""
    <div style='background:#1a1f2e; border-radius:12px; padding:24px; border:1px solid #2a2f3e'>
    <p style='color:#8892a4; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; margin:0 0 12px 0'>HOW IT WORKS</p>
    <div style='display:flex; flex-direction:column; gap:12px'>
    <div style='display:flex; align-items:flex-start; gap:12px'>
      <span style='background:#0f3460; color:#4fc3f7; border-radius:50%; width:26px; height:26px; display:flex; align-items:center; justify-content:center; font-size:0.8rem; font-weight:700; flex-shrink:0; padding-top:2px'>1</span>
      <span style='color:#c9d1d9; font-size:0.95rem'>Download the CSV template and fill in your company financials</span>
    </div>
    <div style='display:flex; align-items:flex-start; gap:12px'>
      <span style='background:#0f3460; color:#4fc3f7; border-radius:50%; width:26px; height:26px; display:flex; align-items:center; justify-content:center; font-size:0.8rem; font-weight:700; flex-shrink:0; padding-top:2px'>2</span>
      <span style='color:#c9d1d9; font-size:0.95rem'>Upload the completed file using the uploader below</span>
    </div>
    <div style='display:flex; align-items:flex-start; gap:12px'>
      <span style='background:#0f3460; color:#4fc3f7; border-radius:50%; width:26px; height:26px; display:flex; align-items:center; justify-content:center; font-size:0.8rem; font-weight:700; flex-shrink:0; padding-top:2px'>3</span>
      <span style='color:#c9d1d9; font-size:0.95rem'>FraudLens runs Beneish M-Score, Z-Score &amp; Isolation Forest scoring</span>
    </div>
    <div style='display:flex; align-items:flex-start; gap:12px'>
      <span style='background:#0f3460; color:#4fc3f7; border-radius:50%; width:26px; height:26px; display:flex; align-items:center; justify-content:center; font-size:0.8rem; font-weight:700; flex-shrink:0; padding-top:2px'>4</span>
      <span style='color:#c9d1d9; font-size:0.95rem'>Download the full PDF investigation report</span>
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
with col_b:
    REQUIRED_COLUMNS = [
        'company_id','company_name',
        'sales_t','sales_t1','COGS_t','COGS_t1',
        'receivables_t','receivables_t1',
        'total_assets_t','total_assets_t1',
        'PPE_t','PPE_t1',
        'net_income_t','net_income_t1',
        'total_debt_t','total_debt_t1',
        'cash_flow_ops_t','SGA_t','SGA_t1',
        'depreciation_t','depreciation_t1'
    ]
    
    template_csv = pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(index=False)
    st.download_button("📥 Download CSV Template", template_csv,
                       file_name="fraudlens_template.csv", mime="text/csv",
                       use_container_width=True)
    
    import io
    template_xlsx_buffer = io.BytesIO()
    pd.DataFrame(columns=REQUIRED_COLUMNS).to_excel(template_xlsx_buffer, index=False)
    template_xlsx_buffer.seek(0)
    st.download_button("📥 Download Excel Template", template_xlsx_buffer,
                       file_name="fraudlens_template.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
st.markdown("---")

# ── File upload ───────────────────────────────────────────────────────────────
st.markdown("""
<p style='color:#8892a4; font-size:0.85rem; text-transform:uppercase; 
letter-spacing:1px; margin:24px 0 8px 0'>UPLOAD YOUR DATA</p>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drop your CSV or Excel file here", type=["csv", "xlsx"],
                                  label_visibility="visible")

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        st.error(f"❌ Missing columns: {missing}")
        st.stop()

    df[REQUIRED_COLUMNS[2:]] = df[REQUIRED_COLUMNS[2:]].fillna(
        df[REQUIRED_COLUMNS[2:]].median())
    st.success(f"✅ {len(df)} companies loaded successfully")

    # ── Beneish M-Score ───────────────────────────────────────────────────────
    def safe_div(a, b):
        return np.where(b != 0, a / b, np.nan)

    df['DSRI'] = safe_div(df['receivables_t']/df['sales_t'],
                           df['receivables_t1']/df['sales_t1'])
    df['GMI']  = safe_div((df['sales_t1']-df['COGS_t1'])/df['sales_t1'],
                           (df['sales_t']-df['COGS_t'])/df['sales_t'])
    df['AQI']  = safe_div(1-(df['PPE_t']/df['total_assets_t']),
                           1-(df['PPE_t1']/df['total_assets_t1']))
    df['SGI']  = safe_div(df['sales_t'], df['sales_t1'])
    df['DEPI'] = safe_div(df['depreciation_t1']/(df['depreciation_t1']+df['PPE_t1']),
                           df['depreciation_t']/(df['depreciation_t']+df['PPE_t']))
    df['SGAI'] = safe_div(df['SGA_t']/df['sales_t'], df['SGA_t1']/df['sales_t1'])
    df['LVGI'] = safe_div(df['total_debt_t']/df['total_assets_t'],
                           df['total_debt_t1']/df['total_assets_t1'])
    df['TATA'] = safe_div(df['net_income_t']-df['cash_flow_ops_t'], df['total_assets_t'])

    df['M_Score'] = (-4.84 + 0.92*df['DSRI'] + 0.528*df['GMI'] +
                     0.404*df['AQI'] + 0.892*df['SGI'] + 0.115*df['DEPI'] -
                     0.172*df['SGAI'] + 4.679*df['TATA'] - 0.327*df['LVGI'])
    df['beneish_flag'] = (df['M_Score'] > -2.22).astype(int)

    # ── Anomaly detection ─────────────────────────────────────────────────────
    df = df.replace([np.inf, -np.inf], np.nan)
    df_scored = df.dropna(subset=['M_Score']).copy()
    FEATURES = ['DSRI','GMI','AQI','SGI','DEPI','SGAI','LVGI','TATA']
    features = df_scored[FEATURES].fillna(0)

    z = np.abs(stats.zscore(features))
    df_scored['z_score_flag'] = (z > 2).any(axis=1).astype(int)
    iso = IsolationForest(contamination=0.2, random_state=42)
    df_scored['iso_flag'] = (iso.fit_predict(features) == -1).astype(int)

    m_min, m_max = df_scored['M_Score'].min(), df_scored['M_Score'].max()
    df_scored['risk_score'] = ((df_scored['M_Score'] - m_min) / (m_max - m_min)
                                if m_max != m_min else 0.5)
    df_scored['risk_score'] = (df_scored['risk_score']
                                + df_scored['z_score_flag']*0.1
                                + df_scored['iso_flag']*0.1).clip(0, 1)

    def classify(s):
        if s >= 0.6: return 'HIGH'
        elif s >= 0.3: return 'MEDIUM'
        return 'LOW'

    df_scored['risk_classification'] = df_scored['risk_score'].apply(classify)
    df_scored['fraud_probability'] = (df_scored['risk_score']*100).round(1)

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Results Summary</div>',
                unsafe_allow_html=True)
    total = len(df_scored)
    high   = (df_scored['risk_classification']=='HIGH').sum()
    medium = (df_scored['risk_classification']=='MEDIUM').sum()
    low    = (df_scored['risk_classification']=='LOW').sum()

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card total">
            <div class="label">TOTAL COMPANIES</div>
            <div class="value">{total}</div>
        </div>
        <div class="metric-card high">
            <div class="label">🔴 HIGH RISK</div>
            <div class="value">{high}</div>
        </div>
        <div class="metric-card medium">
            <div class="label">🟠 MEDIUM RISK</div>
            <div class="value">{medium}</div>
        </div>
        <div class="metric-card low">
            <div class="label">🟢 LOW RISK</div>
            <div class="value">{low}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Risk table ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Full Risk Table</div>',
                unsafe_allow_html=True)

    def highlight_risk(val):
        if val == 'HIGH':   return 'background-color: #4a1f1f; color: #ef9a9a'
        if val == 'MEDIUM': return 'background-color: #3d2a0f; color: #ffcc80'
        if val == 'LOW':    return 'background-color: #1a3a1f; color: #a5d6a7'
        return ''

    display_cols = ['company_id','company_name','M_Score','beneish_flag',
                    'z_score_flag','iso_flag','risk_classification','fraud_probability']
    styled = (df_scored[display_cols]
              .sort_values('fraud_probability', ascending=False)
              .style.map(highlight_risk, subset=['risk_classification'])
              .format({'M_Score': '{:.2f}', 'fraud_probability': '{:.1f}%'}))
    st.dataframe(styled, use_container_width=True, height=400)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Visualisations</div>',
                unsafe_allow_html=True)

    color_map = {'HIGH':'#ef5350','MEDIUM':'#ffa726','LOW':'#66bb6a'}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#1a1f2e')
    for ax in axes:
        ax.set_facecolor('#1a1f2e')
        ax.tick_params(colors='#8892a4')
        ax.xaxis.label.set_color('#8892a4')
        ax.yaxis.label.set_color('#8892a4')
        ax.title.set_color('#ffffff')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2f3e')

    fig.suptitle("FraudLens Risk Analysis", fontsize=14,
                 fontweight='bold', color='#ffffff')

    bar_colors = df_scored['risk_classification'].map(color_map)
    axes[0].bar(range(len(df_scored)), df_scored['M_Score'].values,
                color=bar_colors, edgecolor='#0f1117', linewidth=0.5)
    axes[0].axhline(y=-2.22, color='#4fc3f7', linestyle='--',
                    linewidth=1.5, label='Threshold (-2.22)')
    axes[0].set_title("M-Score by Company", fontweight='bold')
    axes[0].set_xlabel("Company Index")
    axes[0].set_ylabel("Beneish M-Score")
    patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    axes[0].legend(handles=patches, facecolor='#1a1f2e',
                   labelcolor='white', edgecolor='#2a2f3e')

    top15 = df_scored.nlargest(min(15, len(df_scored)), 'fraud_probability')
    axes[1].barh(top15['company_id'], top15['fraud_probability'],
                 color=top15['risk_classification'].map(color_map))
    axes[1].set_title("Top Companies by Fraud Probability", fontweight='bold')
    axes[1].set_xlabel("Fraud Probability (%)")
    axes[1].invert_yaxis()

    counts = df_scored['risk_classification'].value_counts()
    pie_colors = [color_map[k] for k in counts.index]
    axes[2].pie(counts, labels=counts.index, colors=pie_colors,
                autopct='%1.1f%%', startangle=90,
                textprops={'color':'white'})
    axes[2].set_title("Risk Breakdown", fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

    # ── PDF report ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📄 Download Report</div>',
                unsafe_allow_html=True)

    if st.button("🖨️ Generate PDF Report", use_container_width=False):
        with st.spinner("Generating report..."):
            chart_path = tempfile.mktemp(suffix='.png')
            fig.savefig(chart_path, dpi=120, bbox_inches='tight',
                        facecolor='#1a1f2e')
            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4,
                                    topMargin=2*cm, bottomMargin=2*cm,
                                    leftMargin=2*cm, rightMargin=2*cm)
            styles = getSampleStyleSheet()
            title_s = ParagraphStyle('T',parent=styles['Title'],fontSize=16,spaceAfter=6)
            head_s  = ParagraphStyle('H',parent=styles['Heading2'],fontSize=11,spaceAfter=4)
            body_s  = ParagraphStyle('B',parent=styles['Normal'],fontSize=9,
                                      spaceAfter=4,leading=13)
            disc_s  = ParagraphStyle('D',parent=styles['Normal'],fontSize=8,
                                      textColor=colors.grey,spaceAfter=4)
            story = []
            story.append(Paragraph("FraudLens – Fraud Risk Report", title_s))
            story.append(Paragraph(f"Analysis Date: {date.today()} | "
                                   f"Prepared by: Theophilus Abayateye", body_s))
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph("Executive Summary", head_s))
            story.append(Paragraph(
                f"FraudLens analysed {total} company observations. "
                f"{high} ({high/total*100:.1f}%) were classified HIGH risk, "
                f"{medium} ({medium/total*100:.1f}%) MEDIUM, "
                f"and {low} ({low/total*100:.1f}%) LOW.", body_s))
            story.append(Spacer(1,0.3*cm))
            story.append(Paragraph("Risk Visualisations", head_s))
            story.append(RLImage(chart_path, width=16*cm, height=5.5*cm))
            story.append(Spacer(1,0.3*cm))
            story.append(Paragraph("Full Risk Table", head_s))
            tdata = [['Company ID','Company','M-Score','B','Z','IF','Risk','Prob %']]
            for _, row in df_scored.sort_values(
                    'fraud_probability', ascending=False).iterrows():
                tdata.append([
                    str(row['company_id']), str(row['company_name'])[:28],
                    f"{row['M_Score']:.2f}",
                    '✓' if row['beneish_flag']==1 else '',
                    '✓' if row['z_score_flag']==1 else '',
                    '✓' if row['iso_flag']==1 else '',
                    row['risk_classification'],
                    f"{row['fraud_probability']:.1f}"
                ])
            rc = {'HIGH':colors.HexColor('#FADADD'),
                  'MEDIUM':colors.HexColor('#FDEBD0'),
                  'LOW':colors.HexColor('#D5F5E3')}
            t = Table(tdata,
                      colWidths=[2.2*cm,5*cm,1.7*cm,1*cm,1*cm,1*cm,1.7*cm,1.5*cm])
            cmds = [
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2C3E50')),
                ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                ('FONTSIZE',(0,0),(-1,-1),7.5),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#CCCCCC')),
                ('ALIGN',(2,1),(-1,-1),'CENTER'),
                ('TOPPADDING',(0,0),(-1,-1),3),
                ('BOTTOMPADDING',(0,0),(-1,-1),3),
            ]
            for i, row in enumerate(tdata[1:], 1):
                if row[6] in rc:
                    cmds.append(('BACKGROUND',(6,i),(6,i),rc[row[6]]))
            t.setStyle(TableStyle(cmds))
            story.append(t)
            story.append(Spacer(1,0.3*cm))
            story.append(Paragraph("Disclaimer", head_s))
            story.append(Paragraph(
                "This report is for academic and research purposes only. "
                "Classifications do not constitute a finding of fraud or legal "
                "violation. Results should be interpreted alongside qualitative "
                "analysis and professional judgment.", disc_s))
            doc.build(story)
            os.unlink(chart_path)

        st.download_button("📥 Download PDF Report", pdf_buffer.getvalue(),
                           file_name="FraudLens_Report.pdf",
                           mime="application/pdf",
                           use_container_width=False)
        st.success("✅ Report ready!")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4a5568; font-size:0.85rem; padding:16px'>
    🔍 <strong style='color:#8892a4'>FraudLens</strong> · 
    Built by Theophilus Abayateye · 
        <a href='https://github.com/Theo-abaya/FraudLens' 
       style='color:#4fc3f7'>GitHub</a>
</div>
""", unsafe_allow_html=True)