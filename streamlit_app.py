# streamlit_lead_scoring_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="리드 스코어링 대시보드", layout="centered")

st.title("리드 생성 및 평가 자동화 솔루션")
st.markdown("웹 방문, 이메일 클릭, SNS 반응 데이터를 기반으로 리드를 자동 평가합니다.")

# 1. 데이터 업로드
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("업로드된 리드 데이터")
    st.dataframe(df.head())

    if 'converted' not in df.columns:
        st.error("Error: 'converted' 컬럼이 포함되어야 합니다.")
    else:
        # 2. 모델 학습
        X = df.drop(columns=['converted'])
        y = df['converted']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 3. 평가 리포트
        st.subheader("모델 성능 평가")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.json(report)

        # 4. 전체 리드에 대한 스코어링
        df['lead_score'] = model.predict_proba(X)[:, 1]
        st.subheader("리드 스코어링 결과")
        st.dataframe(df.sort_values(by='lead_score', ascending=False).head(10))

        # 5. CSV 다운로드
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8-sig')

        csv = convert_df(df)
        st.download_button(
            label="리드 스코어 결과 다운로드",
            data=csv,
            file_name='scored_leads.csv',
            mime='text/csv',
        )
else:
    st.info("왼쪽 상단에서 예시 CSV 파일을 업로드하세요.")