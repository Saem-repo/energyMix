#%%
import tensorflow as tf
import streamlit as st
from PIL import Image

from random import *

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from streamlit_option_menu import option_menu
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

import os
import sys
import glob

#%%

st.set_page_config(layout='wide',
                   page_icon='./img/smart_grid.png', 
                #    initial_sidebar_state='collapsed',
                   page_title='CEMOS for EnergyMix Decision Support')

st.sidebar.image('./img/SSEL_logo.png')

#%%

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title=None,  # required
                options=["홈",  "커뮤니티 에너지 믹스 데이터 현황", "커뮤니티 에너지 믹스 탐색"],  # required
                # options=["커뮤니티 에너지 믹스 탐색"],  # required
                icons=["house", "search", "list-task"],  # optional
                # icons=["search"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
                styles = {
                "container": {"color":"#000000", "padding": "4!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "green"},
                }
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["홈",  "커뮤니티 에너지 믹스 데이터 현황", "커뮤니티 에너지 믹스 탐색"],  # required
            icons=["house", "sd-card", "search"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles = {
                "container": { "padding": "4!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"}, 
                "nav-link": {"font-size": "25px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "green"},
            }
        )
        return selected


selected = streamlit_menu(example=1)   
    
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.sidebar.markdown('---')

st.sidebar.info(''' **Smart and Sustainable Environment LAB** | **SSEL**     
                    **Contact Info** | 
                    **Call** : +82-42-350-5667   
                    **Website** : <http://ssel.kaist.ac.kr/>   
                ''')


#%%
def home () :
        
    # img = Image.open('./img/home/smart_city.png')
    # st.image(img)
    # st.markdown("---")

    st.markdown(""" <style> .font {
        font-size:45px ; font-family: 'Cooper Black'; color: #0064ff; text-align: center;} 
        </style> """, unsafe_allow_html=True)
    st.markdown(""" <style> .font1 {
        font-size:25px ; font-family: 'Cooper Black'; color: #46CCFF; text-align: center;} 
        </style> """, unsafe_allow_html=True)

    st.markdown(""" <style> .font2 {
        font-size:20px ; font-family: 'Times, Times New Roman, Georgia, serif'; color: #000000; text-align: left;} 
        </style> """, unsafe_allow_html=True)

    st.markdown(""" <style> .font3 {
        font-size:15px ; font-family: 'Times, Times New Roman, Georgia, serif'; color: #FF0000; text-align: left;} 
        </style> """, unsafe_allow_html=True)
    
    
    st.markdown('''<p class="font"><strong>커뮤니티 에너지 믹스 의사결정지원 시스템 Ver. 2 </strong></p>   
                <p class="font1"><strong>CEMOS : Community Energy Mix Optimization System</strong></p>
                ''', unsafe_allow_html=True)
    
    st.markdown("---")
        
    col1, col2, col3 = st.columns([0.7, 0.1, 0.8])
    
    with col1:               # To display the header text using css style
        st.markdown('''
                    - ### 시스템 개요
                        - <p style="font-size:20px;"> 본 시스템은 커뮤니티 유형 및 특성에 따른 에너지 수요와 신재생 에너지원별(풍력, 태영광) LCC를 고려한 최적 커뮤니티 에너지 믹스 체계를 도출하기 위한 목적으로 개발됨 </p>
                    
                    - ### 시스템 기능
                        - <p style="font-size:20px;"> 커뮤니티 에너지(전력, 열 등) 프로필 데이터 현황에 대한 분석 결과 제공 </p>    
                        - <p style="font-size:20px;"> 커뮤니티 특성(위치, 면적, 기후 등)에 따른 다목적 최적화 기반 커뮤니티 에너지 믹스 방안 도출 </p>    
                    ''', unsafe_allow_html=True)
        
        
    with col3:               # To display the header text using css style
        # st.video('./img/home/smartcity.mp4')
        st.image('./img/EnergyMix.png')
    

    st.markdown("---")
    col1, col2, col3 = st.columns([0.15, 0.8, 0.1])
    
    with col2:               
        st.markdown('''<p class="font2"><strong> CEMOS 구조 및 데이터 처리 순서도 </strong></p>  
                       <p class="font3"><strong> ※ 이미지 확대를 원할시 이미지에 마우스 커서를 올렸을 때 나타나는 우측 상단에 확대 버튼을 누르세요.
                    ''', unsafe_allow_html=True)
        st.image('./img/CEMOS.png')
    
    
    # st.markdown(""" <style> .font {
    # text-align: left; font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
    # </style> """, unsafe_allow_html=True)
    # st.markdown('<p class="font">저탄소 에너지효율화 기술 기반 에너지공유 커뮤니티 구축 기술 개발을 위한 리모델링 사례 추천 시스템</p>', unsafe_allow_html=True)    


#%%
def data_summary():

    st.markdown(""" <style> .font {
        font-size:45px ; font-family: 'Cooper Black'; color: #0064ff; text-align: center;} 
        </style> """, unsafe_allow_html=True)

    st.markdown(""" <style> .font2 {
        font-size:20px ; font-family: 'Times, Times New Roman, Georgia, serif'; color: #000000; text-align: left;} 
        </style> """, unsafe_allow_html=True)
    
    st.markdown('<p class="font"><strong>커뮤니티 에너지 믹스 데이터 현황</strong></p>', unsafe_allow_html=True) 
    st.markdown("---")
    
    Energy_profile_elec = pd.read_csv("./data/Energy_profile_gen_elec.csv")
    Energy_profile_gas = pd.read_csv("./data/Energy_profile_gen_gas.csv")

    # Initial price for renewable energy (per kWh)
    ESS_initial = pd.read_csv("./data/ESSPrice.csv")
    PV_initial = pd.read_csv("./data/SolarPrice.csv")
    Wind_initial = pd.read_csv("./data/WindPrice.csv")

    # Diesel generator fuel price (per kWh)
    Diesel_price = pd.read_csv("./data/DieselPrice.csv")

    # Wind turbine generation (per each capacity(10,100,250,750,1500kW))
    Wind_generation = pd.read_csv("./data/WindPower.csv")

    # Environmental cost (for SOx, NOx, Dust, CO2)
    EnvironmentalFee = pd.read_csv("./data/envirper1kW.csv")

    # Hourly electricity cost profile
    Hourly_Elec_Fee = pd.read_csv("./data/ElecPrice.csv")


    profile_cols = st.columns(2)

    with profile_cols[0] :
        st.markdown('<p class="font2"><strong>커뮤니티 전력 데이터 (기본 에너지 프로필 데이터: KAIST 캠퍼스)</strong></p>', unsafe_allow_html=True) 
        profile_elec = Energy_profile_elec.profile_report()
        st_profile_report(profile_elec)
    
    with profile_cols[1] :
        st.markdown('<p class="font2"><strong>커뮤니티 가스 데이터 (기본 에너지 프로필 데이터: KAIST 캠퍼스)</strong></p>', unsafe_allow_html=True) 
        profile_gas = Energy_profile_gas.profile_report()
        st_profile_report(profile_gas)

def cemos():

    st.markdown(""" <style> .font {
        font-size:45px ; font-family: 'Cooper Black'; color: #0064ff; text-align: center;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font"><strong>커뮤니티 최적 에너지 믹스 전략</strong></p>', unsafe_allow_html=True)
    st.markdown('''
                    - ### CEMOS 사용 순서
                        - 커뮤니티 위치 정보 (위도, 경도) 입력
                        - 커뮤니티 규모 (거주 및 업무 시설 연면적) 입력
                        - 각 커뮤니티 지역별 날씨 데이터 업로드
                        - 해당 커뮤니티 최적 에너지 믹스 조건 탐색 
                ''')

    st.markdown("---")

    split_cols = st.columns(2)

    with split_cols[0] :
        print('dddd')


    with split_cols[1] :
        print('Renewal....')

    
page_names_to_funcs = {
    "홈": home,
    "커뮤니티 에너지 믹스 데이터 현황": data_summary,
    "커뮤니티 에너지 믹스 탐색": cemos
    }

page_names_to_funcs[selected]()





