import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==============================================================================
# 1. ตั้งค่าหน้าเว็บและตกแต่ง UI (Custom CSS)
# ==============================================================================
st.set_page_config(
    page_title="Heart Disease KNN Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
    }

    .hero-banner {
        background: linear-gradient(135deg, #EC7063 0%, #C0392B 100%);
        padding: 1.6rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 20px rgba(236, 112, 99, 0.28);
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        color: white;
    }
    .hero-banner p {
        margin: 0.35rem 0 0 0;
        font-size: 0.98rem;
        opacity: 0.95;
    }

    .profile-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 14px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }

    .metric-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
    }
    .metric-title {
        font-size: 0.85rem;
        opacity: 0.8;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #E74C3C;
    }

    .banner-red {
        background: linear-gradient(90deg, #EC7063 0%, #D9534F 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 10px 0 15px 0;
    }
    .banner-blue {
        background: linear-gradient(90deg, #48C9B0 0%, #2E86C1 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 10px 0 15px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. พจนานุกรมคำอธิบายตัวแปร (Mapping Dictionaries)
# ==============================================================================
SEX_MAP = {0: "0 - หญิง (Female)", 1: "1 - ชาย (Male)"}
CHEST_PAIN_MAP = {
    1: "1 - ASY (ไม่มีอาการเจ็บหน้าอก)",
    2: "2 - ATA (เจ็บหน้าอกไม่ชัดเจน/ผิดปกติ)",
    3: "3 - NAP (เจ็บหน้าอกที่ไม่ใช่โรคหัวใจ)",
    4: "4 - TA (เจ็บหน้าอกแบบโรคหัวใจทั่วไป)",
}
FASTING_BS_MAP = {
    0: "0 - ปกติ (≤ 120 mg/dl)",
    1: "1 - น้ำตาลสูง (> 120 mg/dl)",
}
RESTING_ECG_MAP = {
    1: "1 - LVH (กล้ามเนื้อหัวใจห้องล่างซ้ายหนา)",
    2: "2 - Normal (คลื่นไฟฟ้าหัวใจปกติ)",
    3: "3 - ST (คลื่นไฟฟ้าหัวใจ ST-T ผิดปกติ)",
}
EXERCISE_ANGINA_MAP = {
    0: "0 - No (ไม่เจ็บหน้าอกขณะออกกำลังกาย)",
    1: "1 - Yes (เจ็บหน้าอกขณะออกกำลังกาย)",
}
ST_SLOPE_MAP = {
    1: "1 - Down (กราฟ ST ลาดลง)",
    2: "2 - Flat (กราฟ ST แบนราบ)",
    3: "3 - Up (กราฟ ST ลาดขึ้น)",
}

COLS_4 = ["Age", "Sex", "ChestPainType", "MaxHR"]
COLS_11 = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]

# ==============================================================================
# 3. ฟังก์ชันโหลดข้อมูลและแสดงรูปภาพอย่างปลอดภัย
# ==============================================================================
@st.cache_data
def load_dataset():
    for path in ["./data/Heart3.csv", "Heart3.csv"]:
        if os.path.exists(path):
            return pd.read_csv(path), path
    return None, None


def show_safe_image(path, caption="", width=220):
    if os.path.exists(path):
        st.image(path, caption=caption, width=width)
    else:
        st.info(f"🖼️ ไม่พบไฟล์ภาพ: `{path}`")


dt, loaded_path = load_dataset()

if dt is None:
    st.warning("⚠️ ไม่พบไฟล์ `./data/Heart3.csv` หรือ `Heart3.csv` กรุณาอัปโหลดไฟล์ข้อมูลหลักด้านล่าง")
    up_base = st.file_uploader("อัปโหลดไฟล์ Heart3.csv", type=["csv"])
    if up_base is not None:
        dt = pd.read_csv(up_base)
    else:
        st.stop()

# ==============================================================================
# 4. แถบตั้งค่าโมเดล KNN ด้านซ้าย (Sidebar)
# ==============================================================================
with st.sidebar:
    st.header("⚙️ ตั้งค่าโมเดล KNN")
    st.caption("ปรับแต่งพารามิเตอร์เพื่อทดสอบความแม่นยำ")

    feature_mode = st.radio(
        "ชุดตัวแปรที่ใช้สอนโมเดล:",
        options=["4 ตัวแปรหลัก (Age, Sex, CP, MaxHR)", "ครบทั้ง 11 ตัวแปร (ความแม่นยำสูงสุด)"],
        index=1,
    )

    k_val = st.slider(
        "จำนวนเพื่อนบ้าน (K - n_neighbors)",
        min_value=1,
        max_value=29,
        value=7,
        step=2,
        help="แนะนำเลขคี่ เช่น 5, 7, 9 เพื่อป้องกันผลโหวตเสมอกัน",
    )

    weights_opt = st.selectbox(
        "การถ่วงน้ำหนัก (Weights)",
        options=["uniform", "distance"],
        format_func=lambda x: "Uniform (น้ำหนักเท่ากัน)" if x == "uniform" else "Distance (ใกล้กว่ามีน้ำหนักมากกว่า)",
    )

    scaler_opt = st.selectbox(
        "การปรับสเกลข้อมูล (Feature Scaling)",
        options=["StandardScaler", "MinMaxScaler", "RobustScaler", "None (ไม่ปรับสเกล)"],
        index=0,
    )

    clean_zeros = st.checkbox(
        "แทนที่ค่า 0 ผิดปกติ (Cholesterol/RestingBP) ด้วยค่ามัธยฐาน",
        value=True,
    )

# ==============================================================================
# 5. เตรียมข้อมูลและฝึกสอนโมเดล KNN
# ==============================================================================
df_model = dt.copy()
chol_med = df_model.loc[df_model["Cholesterol"] > 0, "Cholesterol"].median()
bp_med = df_model.loc[df_model["RestingBP"] > 0, "RestingBP"].median()

if clean_zeros:
    df_model["Cholesterol"] = df_model["Cholesterol"].replace(0, chol_med)
    df_model["RestingBP"] = df_model["RestingBP"].replace(0, bp_med)

selected_cols = COLS_4 if feature_mode.startswith("4") else COLS_11
X = df_model[selected_cols]
y = df_model["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

if scaler_opt == "StandardScaler":
    scaler = StandardScaler()
elif scaler_opt == "MinMaxScaler":
    scaler = MinMaxScaler()
elif scaler_opt == "RobustScaler":
    scaler = RobustScaler()
else:
    scaler = "passthrough"

# โมเดลสำหรับประเมินผลบนชุดทดสอบ
eval_pipeline = Pipeline(
    [("scaler", scaler), ("knn", KNeighborsClassifier(n_neighbors=k_val, weights=weights_opt))]
)
eval_pipeline.fit(X_train, y_train)
y_pred_test = eval_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test)
rec = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

# โมเดลหลักฝึกสอนจากข้อมูลทั้งหมดสำหรับทำนายเคสใหม่
final_pipeline = Pipeline(
    [("scaler", scaler), ("knn", KNeighborsClassifier(n_neighbors=k_val, weights=weights_opt))]
)
final_pipeline.fit(X, y)

# ==============================================================================
# 6. ส่วนที่ 1: หัวข้อเว็บไซต์และรูปภาพประกอบ 3 คอลัมน์
# ==============================================================================
st.markdown(
    """
    <div class="hero-banner">
        <h1>❤️ Heart Disease Prediction (KNN Classifier)</h1>
        <p>เว็บแอปพลิเคชันจำแนกและพยากรณ์ความเสี่ยงโรคหัวใจด้วยเทคนิค K-Nearest Neighbors</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="profile-card"><h4>💔 Heart Disease</h4>', unsafe_allow_html=True)
    show_safe_image("./img/Heard1.png", caption="ภาวะเสี่ยง / เป็นโรคหัวใจ (1)", width=220)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="profile-card"><h4>💚 No Heart Disease</h4>', unsafe_allow_html=True)
    show_safe_image("./img/Heard2.jpg", caption="หัวใจปกติ / ไม่เป็นโรคหัวใจ (0)", width=220)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="profile-card"><h4>👤 Profile</h4>', unsafe_allow_html=True)
    show_safe_image("./img/Warapon.jpg", caption="ผู้จัดทำโครงงาน", width=220)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# แถบแสดงตัวชี้วัดประสิทธิภาพโมเดล
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">ความแม่นยำ (Accuracy)</div><div class="metric-value">{acc*100:.2f}%</div></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">Precision</div><div class="metric-value">{prec*100:.2f}%</div></div>',
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">Recall</div><div class="metric-value">{rec*100:.2f}%</div></div>',
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">F1-Score</div><div class="metric-value">{f1*100:.2f}%</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

# ==============================================================================
# 7. แบ่งหน้าจอหลักเป็น 3 แท็บใช้งานง่าย
# ==============================================================================
tab_pred, tab_eda, tab_eval = st.tabs(
    [
        "🩺 ทำนายข้อมูลใหม่ (Prediction)",
        "📊 สถิติและจินตทัศน์ข้อมูล",
        "📈 ประสิทธิภาพโมเดล KNN",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: ป้อนข้อมูลใหม่ (รายบุคคล + อัปโหลดไฟล์ CSV ใหม่)
# ------------------------------------------------------------------------------
with tab_pred:
    st.markdown('<div class="banner-blue">🩺 ทำนายความเสี่ยงโรคหัวใจสำหรับชุดข้อมูลใหม่</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "เลือกรูปแบบการป้อนข้อมูล:",
        ["📝 กรอกข้อมูลทีละรายการ (Single Input)", "📂 อัปโหลดไฟล์ CSV ชุดข้อมูลใหม่ (Batch Prediction)"],
        horizontal=True,
    )

    if input_mode == "📝 กรอกข้อมูลทีละรายการ (Single Input)":
        with st.form("single_predict_form"):
            if feature_mode.startswith("4"):
                in_col1, in_col2 = st.columns(2)
                with in_col1:
                    age = st.number_input("กรุณากรอกอายุ (Age) [1 - 120 ปี]", min_value=1, max_value=120, value=45, step=1)
                    sex = st.selectbox("กรุณาเลือกเพศ (Sex)", options=list(SEX_MAP.keys()), index=1, format_func=lambda x: SEX_MAP[x])
                with in_col2:
                    cp = st.selectbox("อาการเจ็บหน้าอก (ChestPainType)", options=list(CHEST_PAIN_MAP.keys()), index=0, format_func=lambda x: CHEST_PAIN_MAP[x])
                    hr = st.number_input("อัตราการเต้นหัวใจสูงสุด (MaxHR) [50 - 250 ครั้ง/นาที]", min_value=50, max_value=250, value=150, step=1)
            else:
                c_a, c_b, c_c = st.columns(3)
                with c_a:
                    age = st.number_input("อายุ (Age - ปี)", min_value=1, max_value=120, value=50, step=1)
                    sex = st.selectbox("เพศ (Sex)", options=list(SEX_MAP.keys()), index=1, format_func=lambda x: SEX_MAP[x])
                    cp = st.selectbox("อาการเจ็บหน้าอก (ChestPainType)", options=list(CHEST_PAIN_MAP.keys()), format_func=lambda x: CHEST_PAIN_MAP[x])
                    ex_angina = st.selectbox("เจ็บหน้าอกขณะออกกำลังกาย (ExerciseAngina)", options=list(EXERCISE_ANGINA_MAP.keys()), format_func=lambda x: EXERCISE_ANGINA_MAP[x])
                with c_b:
                    rest_bp = st.number_input("ความดันโลหิตขณะพัก (RestingBP)", min_value=60, max_value=250, value=130, step=1)
                    chol = st.number_input("คอเลสเตอรอล (Cholesterol)", min_value=0, max_value=700, value=220, step=1)
                    fast_bs = st.selectbox("น้ำตาลในเลือดขณะอดอาหาร (FastingBS)", options=list(FASTING_BS_MAP.keys()), format_func=lambda x: FASTING_BS_MAP[x])
                    hr = st.number_input("อัตราการเต้นหัวใจสูงสุด (MaxHR)", min_value=50, max_value=250, value=145, step=1)
                with c_c:
                    rest_ecg = st.selectbox("ผลคลื่นไฟฟ้าหัวใจ (RestingECG)", options=list(RESTING_ECG_MAP.keys()), format_func=lambda x: RESTING_ECG_MAP[x])
                    oldpeak = st.number_input("ค่าความกดต่ำคลื่น ST (Oldpeak)", min_value=-5.0, max_value=10.0, value=1.0, step=0.1)
                    st_slope = st.selectbox("ความชันกราฟ ST (ST_Slope)", options=list(ST_SLOPE_MAP.keys()), format_func=lambda x: ST_SLOPE_MAP[x])

            submitted = st.form_submit_button("🔍 ทำนายผล", type="primary", use_container_width=True)

        if submitted:
            if feature_mode.startswith("4"):
                x_input = pd.DataFrame([[age, sex, cp, hr]], columns=COLS_4)
            else:
                x_input = pd.DataFrame(
                    [
                        {
                            "Age": age,
                            "Sex": sex,
                            "ChestPainType": cp,
                            "RestingBP": bp_med if (clean_zeros and rest_bp == 0) else rest_bp,
                            "Cholesterol": chol_med if (clean_zeros and chol == 0) else chol,
                            "FastingBS": fast_bs,
                            "RestingECG": rest_ecg,
                            "MaxHR": hr,
                            "ExerciseAngina": ex_angina,
                            "Oldpeak": oldpeak,
                            "ST_Slope": st_slope,
                        }
                    ]
                )

            pred = final_pipeline.predict(x_input)[0]
            proba = final_pipeline.predict_proba(x_input)[0]
            risk_pct = proba[1] * 100

            r_col1, r_col2, r_col3 = st.columns([1.2, 1, 1])

            with r_col1:
                st.subheader("สรุปผลการทำนาย:")
                if pred == 0:
                    st.success("✅ **ผลการทำนาย: 0 (ปกติ / ไม่เสี่ยงโรคหัวใจ)**")
                    st.write(f"ความมั่นใจผลปกติ: **{proba[0]*100:.1f}%** (ความเสี่ยงโรคหัวใจ {risk_pct:.1f}%)")
                else:
                    st.error("⚠️ **ผลการทำนาย: 1 (มีความเสี่ยงเป็นโรคหัวใจ)**")
                    st.write(f"โอกาสเสี่ยงเป็นโรคหัวใจ: **{risk_pct:.1f}%**")

            with r_col2:
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=risk_pct,
                        number={"suffix": "%"},
                        title={"text": "ระดับความเสี่ยง"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#E74C3C" if pred == 1 else "#2ECC71"},
                            "steps": [
                                {"range": [0, 40], "color": "#D5F5E3"},
                                {"range": [40, 70], "color": "#FCF3CF"},
                                {"range": [70, 100], "color": "#FADBD8"},
                            ],
                        },
                    )
                )
                fig_gauge.update_layout(height=230, margin=dict(l=15, r=15, t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with r_col3:
                if pred == 0:
                    show_safe_image("./img/Heard2.jpg", caption="ปกติ (No Heart Disease)", width=210)
                else:
                    show_safe_image("./img/Heard1.png", caption="เสี่ยงโรคหัวใจ (Heart Disease)", width=210)
        else:
            st.info("👆 กรุณากรอกข้อมูลด้านบนให้ครบถ้วน แล้วกดปุ่ม **'🔍 ทำนายผล'**")

    else:
        st.write(f"อัปโหลดไฟล์ CSV ที่มีคอลัมน์ครบตามโหมดที่เลือก: `{', '.join(selected_cols)}`")

        sample_csv = dt[selected_cols].head(5).to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 ดาวน์โหลดไฟล์ตัวอย่าง (sample_input.csv)",
            data=sample_csv,
            file_name="sample_input.csv",
            mime="text/csv",
        )

        batch_file = st.file_uploader("เลือกไฟล์ CSV ที่ต้องการทำนายผล", type=["csv"], key="batch_csv")
        if batch_file is not None:
            try:
                new_df = pd.read_csv(batch_file)
                if new_df.empty:
                    st.error("❌ ไฟล์ที่อัปโหลดไม่มีข้อมูลอยู่ภายใน กรุณาตรวจสอบไฟล์อีกครั้ง")
                else:
                    missing = [c for c in selected_cols if c not in new_df.columns]
                    if missing:
                        st.error(f"❌ ไฟล์ขาดคอลัมน์ที่จำเป็น: {missing}")
                    else:
                        x_batch = new_df[selected_cols].copy()
                        if clean_zeros and "Cholesterol" in x_batch.columns:
                            x_batch["Cholesterol"] = x_batch["Cholesterol"].replace(0, chol_med)
                        if clean_zeros and "RestingBP" in x_batch.columns:
                            x_batch["RestingBP"] = x_batch["RestingBP"].replace(0, bp_med)

                        preds = final_pipeline.predict(x_batch)
                        probs = final_pipeline.predict_proba(x_batch)[:, 1] * 100

                        out_df = new_df.copy()
                        out_df["Predicted_HeartDisease"] = preds
                        out_df["Result_Label"] = np.where(preds == 1, "เสี่ยงโรคหัวใจ (1)", "ปกติ (0)")
                        out_df["Risk_Probability (%)"] = np.round(probs, 2)

                        st.success(f"✅ ทำนายข้อมูลใหม่สำเร็จทั้งหมด {len(out_df)} รายการ!")
                        st.dataframe(out_df, use_container_width=True)

                        res_csv = out_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "💾 ดาวน์โหลดผลลัพธ์การทำนาย (CSV)",
                            data=res_csv,
                            file_name="knn_prediction_results.csv",
                            mime="text/csv",
                            type="primary",
                        )
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")

# ------------------------------------------------------------------------------
# TAB 2: ตารางข้อมูลและสถิติค่าเฉลี่ย (ส่วนที่ 2 ของ Home.py เดิม)
# ------------------------------------------------------------------------------
with tab_eda:
    st.markdown('<div class="banner-red">📊 สถิติข้อมูลโรคหัวใจ (Heart3.csv)</div>', unsafe_allow_html=True)

    st.subheader("ตัวอย่างชุดข้อมูล (10 แถวแรก)")
    st.dataframe(dt.head(10), use_container_width=True)

    dx2 = pd.DataFrame(
        [dt["Age"].mean(), dt["Sex"].mean(), dt["ChestPainType"].mean(), dt["MaxHR"].mean()],
        index=["Age", "Sex", "ChestPainType", "MaxHR"],
        columns=["ค่าเฉลี่ย"],
    )

    if st.button("แสดงการจินตทัศน์ข้อมูล (Bar Chart)", type="primary"):
        st.bar_chart(dx2, color="#EC7063")
    else:
        st.caption("กดปุ่มด้านบนเพื่อแสดงกราฟค่าเฉลี่ยของตัวแปรหลัก")

# ------------------------------------------------------------------------------
# TAB 3: กราฟเปรียบเทียบค่า K และ Confusion Matrix
# ------------------------------------------------------------------------------
with tab_eval:
    c_k1, c_k2 = st.columns([1.4, 1])

    with c_k1:
        st.subheader("เปรียบเทียบความแม่นยำตามค่า K (K = 1 ถึง 29)")
        k_list = list(range(1, 30, 2))
        test_accs = []
        cv_accs = []

        for k in k_list:
            pipe_k = Pipeline([("scaler", scaler), ("knn", KNeighborsClassifier(n_neighbors=k, weights=weights_opt))])
            cv_accs.append(cross_val_score(pipe_k, X, y, cv=5).mean() * 100)
            pipe_k.fit(X_train, y_train)
            test_accs.append(pipe_k.score(X_test, y_test) * 100)

        df_k = pd.DataFrame({"ค่า K": k_list, "Test Accuracy (%)": test_accs, "5-Fold CV Accuracy (%)": cv_accs})
        fig_k = px.line(df_k, x="ค่า K", y=["Test Accuracy (%)", "5-Fold CV Accuracy (%)"], markers=True)
        fig_k.add_vline(x=k_val, line_dash="dash", line_color="green", annotation_text=f"K={k_val}")
        fig_k.update_layout(height=360)
        st.plotly_chart(fig_k, use_container_width=True)

    with c_k2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_test)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["ทำนาย: ปกติ (0)", "ทำนาย: โรคหัวใจ (1)"],
            y=["จริง: ปกติ (0)", "จริง: โรคหัวใจ (1)"],
            color_continuous_scale="Reds",
        )
        fig_cm.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)