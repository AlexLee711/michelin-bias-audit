import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import streamlit.components.v1 as components
from PIL import Image
import os

# --- Page Configuration ---
st.set_page_config(page_title="Michelin Bias Analysis", layout="wide", page_icon="🍴")

# 💡 Ultimate anti-jitter CSS: Hiding the fullscreen button
st.markdown(
    """
    <style>
        button[title="View fullscreen"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌟 Michelin Star Fairness Engine: Full Report")
st.markdown("Developed by **Eddie & Alex** | Quantifying Geospatial Bias")

# --- Sidebar Navigation ---
st.sidebar.header("📁 Project Navigation")
phase = st.sidebar.radio("Select Phase:", [
    "Phase 1 & 2: Data Exploration (EDA)",
    "Phase 3: Model & SHAP Explainability",
    "Phase 4: Fairness Demo & USA Comparison",
    "Phase 5: Case Study - The Clustering Bias",
    "Phase 6: Case Study - The Cultural Gap"
])

# ==========================================
# Section 1: Phase 1 & 2 (Data Exploration)
# ==========================================
if phase == "Phase 1 & 2: Data Exploration (EDA)":
    st.header("📊 Phase 1 & 2: Descriptive Analysis")
    st.subheader("🌍 1. Global Michelin Award Distribution")
    
    file_path = 'michelin_feature_engineered_v3.csv'
    
    # 1. 先檢查檔案到底存不存在
    if not os.path.exists(file_path):
        st.error(f"🚨 ERROR: 找不到檔案 '{file_path}'！請檢查檔案是否有上傳，且大小寫完全一致。")
    else:
        # 2. 如果檔案存在，嘗試畫圖並捕捉真實的錯誤訊息
        try:
            df = pd.read_csv(file_path)
            
            # 確保經緯度資料格式正確 (避免字串導致 Folium 崩潰)
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            df_map = df.dropna(subset=['latitude', 'longitude'])
            
            heat_data = [[row['latitude'], row['longitude'], float(row['star_level'])] for index, row in df_map.iterrows() if row['star_level'] > 0]
            
            m = folium.Map(location=[df_map['latitude'].mean(), df_map['longitude'].mean()], zoom_start=2, tiles="CartoDB positron")
            HeatMap(heat_data, radius=12, blur=15, gradient={0.4: 'blue', 0.65: 'lime', 1.0: 'red'}).add_to(m)
            
            components.html(m._repr_html_(), height=450)
            
        except Exception as e:
            # 將真正的系統錯誤印在網頁上給我們看
            st.error(f"🚨 系統錯誤 (System Error): {e}")

    st.divider()

    st.subheader("📉 2. Bias Matrices: Star Level vs. Recognition")
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists('star_bias_matrix.png'): st.image('star_bias_matrix.png', caption="Star Bias Matrix")
    with c2:
        if os.path.exists('recog_bias_matrix.png'): st.image('recog_bias_matrix.png', caption="Recognition Bias Matrix")

    st.divider()
    st.subheader("🍲 3. Native vs. Foreign Cuisine Analysis")
    if os.path.exists('native_comparison.png'): st.image('native_comparison.png', caption="Cuisine Type Impact")

# ==========================================
# Section 2: Phase 3 (SHAP side-by-side)
# ==========================================
elif phase == "Phase 3: Model & SHAP Explainability":
    st.header("🤖 Phase 3: Machine Learning Explainability")
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists('shap_star.png'): st.image('shap_star.png', caption="The Bias of Getting STARS")
    with col2:
        if os.path.exists('shap_recog.png'): st.image('shap_recog.png', caption="The Logic of RECOGNITION")
    st.info("💡 **Key Insight:** Notice how 'is_global_hub' has a massive positive tail for Stars, but almost no impact for Recognition.")

# ==========================================
# Section 3: Phase 4 (USA Comparison & Simulator)
# ==========================================
elif phase == "Phase 4: Fairness Demo & USA Comparison":
    st.header("🇺🇸 Phase 4: Regional Bias in the USA")
    st.markdown("If Michelin is purely about the food, location should not matter. Let's test this using model residuals across the United States.")

    st.info("📊 **What is a 'Residual'?** It is the difference between what our ML model predicts a restaurant *should* receive and what Michelin *actually* gave them. A positive residual means the region is systematically **under-rated**. A negative residual means it is **over-rated** (favored).")

    st.subheader("📍 Level 1: California vs. New York (State-Level)")
    st.caption("Sample size: CA (N=893) vs. NY (N=404).")

    col_ca, col_ny, col_gap = st.columns(3)
    col_ca.metric("CA Avg Residual", "+0.053", "Under-rated Penalty", delta_color="inverse")
    col_ny.metric("NY Avg Residual", "+0.002", "Highly Accurate", delta_color="off")
    col_gap.metric("The State Gap", "0.051", "NY Advantage", delta_color="inverse")

    st.divider()

    st.subheader("🌊 Level 2: West Coast vs. East Coast (Macro-Level)")
    c1, c2, c3 = st.columns(3)
    c1.metric("West Coast Residual", "+0.008", "Systemic Penalty", delta_color="inverse")
    c2.metric("East Coast Residual", "-0.019", "Favored / Over-rated", delta_color="normal")
    c3.metric("The Coastal Gap", "0.028", "East Coast Advantage", delta_color="inverse")

    st.success("💡 **Insight: The East Coast Favoritism.** Michelin launched its first US guide in New York in 2005. Our data reveals a lingering geographic bias: NY is rated almost exactly as the model expects (residual ~0), while California and the West Coast face a systemic geographical penalty.")

    st.divider()

    st.subheader("🎛️ Real-Time Bias Simulator")
    st.markdown("Observe how simply checking the 'East Coast / Global Hub' box changes a restaurant's predicted destiny.")
    c_in, c_out = st.columns([1, 1])
    with c_in:
        p = st.slider("Price Level ($)", 1, 4, 3)
        dens = st.slider("Nearby Michelin Density (Clustering)", 0, 300, 50)
        is_hub = st.checkbox("Relocate to NY / East Coast Hub")
    with c_out:
        hub_bonus = 0.051 if is_hub else 0
        star_val = (p * 0.35) + hub_bonus + (dens * 0.001)
        st.metric("Predicted Star Potential", f"{min(3.0, star_val):.2f} / 3.0 Stars", delta=f"+{hub_bonus:.3f} Geographic Bonus" if is_hub else "No Bonus")

        if is_hub:
            st.warning("🚨 Verdict: Location advantage detected. This restaurant is riding the East Coast inflation.")
        else:
            st.info("⚖️ Verdict: Standard evaluation. This restaurant faces the West Coast penalty.")

# ==========================================
# Section 4: Phase 5 (Cluster Case Study)
# ==========================================
elif phase == "Phase 5: Case Study - The Clustering Bias":
    st.header("🌌 Phase 5: The Gravity of Stars (Clustering Bias)")
    st.markdown("Are Michelin stars solitary geniuses, or do they form exclusive culinary ecosystems?")
    st.divider()
    st.subheader("📊 1. Probability of Proximity: The 5km/10km Rule")
    if os.path.exists('cluster_probability.png'): st.image('cluster_probability.png', use_container_width=True)
    st.divider()
    st.subheader("🎯 2. The Exclusive Privilege of Clustering")
    if os.path.exists('cluster_average_hub.png'):
        st.image('cluster_average_hub.png', use_container_width=True)
        st.success("💡 **Conclusion:** Michelin's 'clustering' phenomenon is almost entirely an exclusive privilege granted to Global Hubs.")

# ==========================================
# Section 5: Phase 6 (Cultural Gap Case Study)
# ==========================================
elif phase == "Phase 6: Case Study - The Cultural Gap":
    st.header("🗾 Phase 6: The Cultural Gap (Michelin vs. Tabelog)")
    st.markdown("We cross-referenced our Tokyo dataset (N=162) with **Tabelog**, Japan's ultra-strict local rating platform, to see if the 'Western Expert' agrees with the 'Local Crowd'.")

    st.divider()

    # Data Snapshot
    st.subheader("📋 Data Snapshot: The Baseline Truth")
    st.markdown("Hard metrics show that standard deviation (Std) shrinks as star levels increase.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("1-Star Avg (N=124)", "3.85 / 5.0", "Std: ±0.27", delta_color="off")
    col2.metric("2-Star Avg (N=26)", "4.07 / 5.0", "Std: ±0.28", delta_color="off")
    col3.metric("3-Star Avg (N=12)", "4.32 / 5.0", "Std: ±0.21", delta_color="off")
    col4.metric("Correlation (Pearson)", "0.453", "Moderate Positive", delta_color="normal")

    st.divider()

    st.subheader("🚪 1. Expert vs Crowd: The Pyramid of Exclusivity")
    if os.path.exists('tabelog_star_boxplot.png'):
        st.image('tabelog_star_boxplot.png', use_container_width=True)
        st.info("💡 **Insight:** There is a moderate positive correlation (**Pearson: 0.453**). While 1-Star restaurants have a chaotic spread, 3-Star restaurants are tightly compressed at the top (4.2+).")
    else:
        st.warning("tabelog_star_boxplot.png not detected")

    st.divider()

    st.subheader("📉 2. The Star Illusion: Does More Stars = Higher Local Score?")
    if os.path.exists('stars_vs_score_trend.png'):
        st.image('stars_vs_score_trend.png', use_container_width=True)
        st.success("💡 **Conclusion: The Sushi Death Cross.** As Sushi is promoted from 1 Star to 2 Stars by Michelin, its local Tabelog score actually **drops**. Michelin's criteria for 'upgrades' may be viewed negatively by local purists who value traditional experiences.")
    else:
        st.warning("stars_vs_score_trend.png not detected")