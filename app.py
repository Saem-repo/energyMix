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
import streamlit_nested_layout

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
                options=["홈",  "커뮤니티 에너지 믹스 데이터 현황", "커뮤니티 에너지 믹스 탐색(신규)", "커뮤니티 에너지 믹스 탐색(기존)"],  # required
                # options=["커뮤니티 에너지 믹스 탐색"],  # required
                icons=["house", "search", "list-task", "list-task"],  # optional
                # icons=["search"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
                # orientation="horizontal",
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



def cemos_new():

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
        st.markdown('''
                        <p class="font2"><strong> CEMOS 입력 정보 </strong></p>   
                    ''', unsafe_allow_html=True)

        st.markdown('1. 커뮤니티 위치 정보* (입력필수)')

        community_cols = st.columns(1)

        with community_cols[0] :
            # uploaded_file = st.file_uploader("날씨 데이터 업로드", type = ['csv'])

            def format_func(dict, option):
                return dict[option]

            community_loc = {1: '강원', 2: '경기', 3: '경남', 4: '경북', 5: '광주', 6: '대구', 7: '대전',
                        8: '부산', 9: '서울', 10: '세종', 11: '울산', 12: '인천', 13: '전남', 14: '전북',
                        15: '제주', 16: '충남', 17: '충북'}

            community_loc_info = st.selectbox('커뮤니티 개발 게획을 위한 지역을 선택하세요', options = list(community_loc.keys()), format_func=lambda x: community_loc[x])
            
            
            community_cols = ['Loc', 'solar_thermal', 'solar', 'wind', 'hydro', 'geo_thermal',
                            'water_thermal', 'bio', 'fuel_cell']

            # community_df_2019 = pd.read_csv("./data/energy_gen/2019_"+"energy_gen"+".csv", encoding='cp949')
            # community_df_2020 = pd.read_csv("./data/energy_gen/2020_"+"energy_gen"+".csv", encoding='cp949')
            community_df_2021 = pd.read_csv("./data/energy_gen/2021_"+"energy_gen"+".csv", encoding='cp949')

            if community_loc_info == 1 : 
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "강원", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0

            elif community_loc_info == 2 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "경기", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0

            elif community_loc_info == 3 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "경남", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0

            elif community_loc_info == 4 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "경북", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0

            elif community_loc_info == 5 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "광주", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0

            elif community_loc_info == 6 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "대구", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0

            elif community_loc_info == 7 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "대전", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0

            elif community_loc_info == 8 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "부산", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0
            
            elif community_loc_info == 9 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "서울", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0

            elif community_loc_info == 10 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "세종", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0

            elif community_loc_info == 11 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "울산", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0

            elif community_loc_info == 12 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "인천", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0

            elif community_loc_info == 13 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "전남", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0

            elif community_loc_info == 14 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "전북", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0
            
            elif community_loc_info == 15 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "제주", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0

            elif community_loc_info == 16 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "충남", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0

            elif community_loc_info == 17 :
                energy_gen = community_df_2021.loc[community_df_2021['Loc'] == "충북", :]
                energy_gen = energy_gen.loc[:,['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell']]
                energy_gen['ess'] = 0
            
        st.markdown("---")
        
        st.markdown('2. 커뮤니티 규모 정보-건축물 유형별 전체 연면적* (입력필수)')
        com_scale_cols = st.columns(4)

        with com_scale_cols[0] :
            com_area_off = st.text_input("업무시설 전체 연면적(m2)", "", max_chars=100, placeholder="전체 연면적을 입력해주세요")
        
        with com_scale_cols[1] :
            com_area_res = st.text_input("거주시설 전체 연면적(m2)", "", max_chars=100, placeholder="전체 연면적을 입력해주세요")

        # with com_scale_cols[2] :
        #     com_area_etc = st.text_input("기타시설 전체 연면적(m2)", "", max_chars=100, placeholder="전체 연면적을 입력해주세요")
        
        with com_scale_cols[2] :
            com_area_etc = st.text_input("거주 총 인원(명)", "", max_chars=100, placeholder="전체 수용인원을 입력해주세요")

        # with com_scale_cols[3] :
        #     com_occ = st.text_input("거주 총 인원(명)", "", max_chars=100, placeholder="전체 수용인원을 입력해주세요")
            
        st.markdown("---")

        st.markdown('3. 선호 신재생 에너지원* (입력필수)')
        
        renew_energies = {1: '태양광', 2: '태양열', 3: '지열', 4: '수열', 5: '풍력', 6: '수력', 7: '수소'}

        def format_func(dict, option):
                return dict[option]

        renew_info = st.multiselect('선호 신재생 에너지원을 선택하세요', options = list(renew_energies.keys()), format_func=lambda x: renew_energies[x])
        # st.write(renew_info)


        st.markdown("---")
        
        st.markdown('4. 에너지 믹스 평가 기준-비용, 에너지발전효율성, 탄소배출량 중 하나* (입력필수)')
        
        # eval_idx = {1: '비용', 2: '탄소배출량(Co2)', 3: '안전성', 4: '안정성', 5: '환경부하'}
        eval_idx = {1: '비용', 2: '에너지발전효율성', 3: '탄소배출량(Co2)'}
        eval_score = st.multiselect('선호하는 평가 기준을 선택하세요', options = list(eval_idx.keys()), format_func=lambda x: eval_idx[x])

        # eval_score = st.radio("선호하는 평가 기준을 선택하세요",
        # ["LCC", "탄소배출량(Co2)", "안전성", "안정성", "환경부하"],
        # horizontal=True)
        st.write(eval_score)

        st.markdown("---")

        st.markdown('5. 커뮤니티 연간 총 에너지 부하(전기/열) - 임시')
        
        com_energy_cols = st.columns(1)

        with com_energy_cols[0] :
            total_com_load = st.text_input("커뮤니티 총 에너지 부하를 입력해주세요", "", max_chars=100, placeholder="")
        
        
        gen_flag = 0
                
        if com_area_off == "" and com_area_res == "" and com_area_etc == "" :
            st.error("필수 입력 정보를 입력해주세요")
    
        else :
                # if uploaded_file is not None :
            #     weather_df = pd.read_csv(uploaded_file, encoding='cp949')
            if st.button("최적 에너지믹스 구성 도출") :

                with st.spinner('커뮤니티 에너지믹스 계산 중'):

                    import numpy as np
                    from platypus.problems import Problem, Real
                    from platypus.algorithms import NSGAII, NSGAIII
                    from platypus.core import Constraint, nondominated

                    # 태양열, 태양광, 풍력, 지열, 수열, 연료전지, ESS 시스템의 설치 비용 (원/kW)
                    install_cost = np.array([2000000, 1500000, 2522000, 15000000, 10000000, 15558000, 2000000])

                    # 태양열, 태양광, 풍력, 지열, 수열, 연료전지, ESS 시스템 시스템의 운영 비용 (원/kW)
                    operating_cost = np.array([100000, 100000, 2000000, 200000, 150000, 1000000, 500000])

                    # 신재생 에너지원별 발전 잠재량 (MWh/year)
                    # generation_potential = np.array([8234040, 204827560, 1000180, 57091670, 0, 8722500, 0])

                    # 태양광 고정식 
                    # 신재생 에너지 생산량 = 원별 설치규모 * 단위 에너지 생산량 * 원별 보정계수
                    # 태양열(평판형): 1060.88 태양광(고정식): 1290.1 풍력9소형): 10687.5, 지열(수직밀폐): 1088.64 수열: 1123.2
                    # 연료전지  (PEMFC): 16313

                    generation_potential = np.array([1060.88, 1290.1, 10687.5, 1088.64, 1123.2, 16313, 0])

                    # 커뮤니티의 전기 수요 (kWh/year)
                    community_demand = int(total_com_load)

                    # ESS 용량 범위 (kWh)
                    ess_range = (0, community_demand*0.2)

                    # 최소 신재생 에너지 공급 양도 (kWh/year)
                    min_renewable_demand = 0.32 * community_demand

                    def objective(variables):
                        # ESS 용량을 제외한 설치용량과 발전 잠재량을 곱하여 실제 발전량을 계산합니다.
                        actual_generation = variables[:-1] * generation_potential[:-1]
                        
                            
                        # ESS 용량을 더하여 전체 발전량을 계산합니다.
                        # total_generation = np.sum(actual_generation) + variables[-1]
                        total_generation = np.sum(actual_generation)

                        # print(variables[:-1], variables[-1])

                        total_cost = np.sum(variables[:-1] * install_cost[:-1]) + np.sum(actual_generation * operating_cost[:-1])
                        
                        
                        # 커뮤니티의 전기 수요와 전체 발전량을 비교하여 비용의 크기를 조정합니다.
                        # 전체 발전량이 커뮤니티의 전기 수요보다 작은 경우, 발전량의 부족분에 대한 비용을 추가합니다.
                        # 전체 발전량이 커뮤니티의 전기 수요보다 큰 경우, 초과된 발전량에 대한 비용을 추가합니다.
                        if total_generation < community_demand:
                            total_cost = np.sum(variables[:-1] * install_cost[:-1]) + np.sum(actual_generation * operating_cost[:-1]) + (community_demand - total_generation) * operating_cost.mean()
                        else:
                            total_cost = np.sum(variables[:-1] * install_cost[:-1]) + np.sum(actual_generation * operating_cost[:-1]) + (total_generation - community_demand) * operating_cost.mean() * 5

                        # print(total_cost, total_generation)
                        
                        # 최소 신재생 에너지 공급 양도 제한조건 추가
                        if np.sum(actual_generation) < min_renewable_demand:
                            total_cost += (min_renewable_demand - np.sum(actual_generation)) * operating_cost.mean() * 2

                        
                        # ESS 용량의 설치 비용과 운영 비용을 계산합니다.
                        ess_install_cost = variables[-1] * install_cost[-1]
                        ess_operating_cost = variables[-1] * operating_cost[-1]

                        # 전체 설치비용과 운영비용에 ESS 용량의 비용을 추가합니다.
                        total_cost += ess_install_cost + ess_operating_cost

                        
                        return [total_cost, -total_generation]

                        return [total_cost, -total_generation]

                    problem = Problem(len(install_cost), 2)

                    # problem.types[:-1] = Real(0, community_demand / np.sum(generation_potential))

                    problem.types[0] = Real(0, community_demand / generation_potential[0])
                    problem.types[1] = Real(0, community_demand / generation_potential[1])
                    problem.types[2] = Real(0, community_demand / generation_potential[2])
                    problem.types[3] = Real(0, community_demand / generation_potential[3])
                    problem.types[4] = Real(0, community_demand / generation_potential[4])
                    problem.types[5] = Real(0, community_demand / generation_potential[5])

                    problem.types[-1] = Real(ess_range[0], ess_range[1])
                    problem.function = objective

                    problem.directions = [Problem.MINIMIZE, Problem.MAXIMIZE]

                    algorithm = NSGAII(problem)
                    algorithm.run(20000)

                    nondominated_sol = nondominated(algorithm.result)

                    opt_var_list = []
                    opt_ess_list = []
                    opt_energy_gen_list = []
                    opt_cost_list = []


                    for i in range(30):
                        # print(f"최적의 신재생 에너지 설치용량: {nondominated_sol[i].variables[:-1]}")
                        # print(f"ESS 용량: {nondominated_sol[i].variables[-1]:.2f} kW")
                        # print(f"최적의 발전량: {-nondominated_sol[i].objectives[1]:.2f} kWh/year")
                        # print(f"총 비용: {nondominated_sol[i].objectives[0]:.2f} 원")
                        # print("==================================")

                        opt_var_list.append(nondominated_sol[i].variables[:-1])
                        opt_ess_list.append(nondominated_sol[i].variables[-1])
                        opt_energy_gen_list.append(-nondominated_sol[i].objectives[1])
                        opt_cost_list.append(nondominated_sol[i].objectives[0])  

                    st.success('Finished') 

                    # 여기서부터는 데이터프레임으로 만들어야함... 플랏 시켜야 되서
                    
                    # 데이터 컬럼명
                    cols = ['solar_thermal','solar','wind','geo_thermal','water_thermal','fuel_cell', 'ess', 'opt_energy', 'opt_cost']

                    df_var = pd.DataFrame(opt_var_list)
                    df_ess = pd.DataFrame(opt_ess_list)
                    df_energy_gen = pd.DataFrame(opt_energy_gen_list)
                    df_cost = pd.DataFrame(opt_cost_list)

                    df_total = pd.concat([df_var, df_ess, df_energy_gen, df_cost], axis=1)

                    df_total.columns = cols
                    opt_can_df = df_total.sort_values('opt_cost', ascending=True).iloc[0:10,:]

                    

                    with split_cols[1] :
                        st.markdown('''
                                            <p class="font2"><strong> CEMOS 출력 정보 </strong></p>   
                                    ''', unsafe_allow_html=True)

                        table_cols = st.columns(1)

                        with table_cols[0]:               
                            st.markdown('''
                                            <p class="font2"><strong> 커뮤니티 에너지 믹스 구성을 위한 다목적 최적화(비용,에너지 발전량) 결과  </strong></p>   
                                        ''', unsafe_allow_html=True)
                            kor_cols = ['태양열','태양광','풍력','지열','수열','연료전지', 'ESS', '최적_에너지발생량', '최적_비용']
                            opt_can_df.columns = kor_cols
                            st.dataframe(opt_can_df)

                        # 결과들은 총 3개 그래피 및 1개 테이블로 시각화
                        # 바 차트 그래프 1,4 파이 차트 그래프 3
                        # 결과에 대한 테이블 1

                        graph_cols_1 = st.columns(1)

                        st.markdown(""" <style> .font2 {
                        font-size:20px ; font-family: 'Times, Times New Roman, Georgia, serif'; color: #000000; text-align: left;} 
                        </style> """, unsafe_allow_html=True)

                        import matplotlib.font_manager as fm

                        font_name = fm.FontProperties(fname="./font/Malgun Gothic.ttf").get_name()
                        font = fm.FontProperties(fname="./font/Malgun Gothic.ttf")
                        plt.rc('font', family=font_name)

                        plt.rcParams['font.family'] ='Malgun Gothic'
                        plt.rcParams['axes.unicode_minus'] =False


                        with graph_cols_1[0] :
                            # st.write("커뮤니티 연간 에너지 발전량")
                            st.markdown('<p class="font2"><strong>최적 에너지 믹스 구성 비율</strong></p>', unsafe_allow_html=True)
                            
                            mix_ratio = []

                            for i in range(7) :
                                mix_ratio.append(np.round(df_total.sort_values('opt_energy', ascending=False).iloc[0,i], 3)/np.sum(np.round(df_total.sort_values('opt_energy', ascending=False).iloc[0,:-2],2)))

                            pie_df = pd.DataFrame(np.round(mix_ratio, 3)*100)
                            pie_df = pie_df.T
                            pie_df.columns = kor_cols[:-2]

                            
                            temp = pie_df.values
                            
                            exp = [0, 0.4, 0.5]
                            fig = plt.figure(figsize=(10,9))
                            plt.rc('font', family = 'Malgun Gothic' )
                            # plt.pie((temp/total_sum)*100, labels = result_df_energy.iloc[:,6:9].columns,
                            #          autopct='%.2f%%', explode=exp, rotatelabels=True)

                            plt.pie(temp.ravel(), pctdistance=1.15, autopct='%.2f%%', explode = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
                                        textprops={'fontsize': 15})
                            plt.legend(loc='upper right', labels = pie_df.columns, fontsize=20, prop=font)

                            st.pyplot(fig)

                        graph_cols_2 = st.columns(1)
                        with graph_cols_2[0] :
                            st.markdown('<p class="font2"><strong>최적 에너지 믹스 성능 평가</strong></p>', unsafe_allow_html=True)
                            
                            top5_opt_performance = np.round(df_total.sort_values('opt_energy', ascending=False).iloc[0:5,:],3)
                            
                            top5_opt_performance = top5_opt_performance.iloc[:,-2:]

                            top5_opt_performance['opt_cost'] = top5_opt_performance['opt_cost']/1000000
                            top5_opt_performance['opt_energy'] = top5_opt_performance['opt_energy']

                            top5_opt_performance.index = ['Sol_1','Sol_2','Sol_3','Sol_4','Sol_5']
                            top5_opt_performance.columns = ['에너지발전효율(Kwh)','연간에너지비용(백만원)']

                            # fig = plt.figure(figsize=(13,6))
                            # ax = fig.add_subplot(111)

                            # rects = plt.barh(bar_graph_2_df.columns, bar_graph_2_df.iloc[0,:], color=['r','g','b'], align='center', height=0.5)
                            
                            # top5_opt_performance.plot(kind='barh')
                            # plt.yticks(ypos, industry)

                            # for i, rect in enumerate(rects):
                            #     ax.text(1 * rect.get_width(), rect.get_y() + rect.get_height() / 2.0, round(bar_graph_2_df.iloc[0,:][i], 2), fontsize=13)

                            plt.xticks(fontsize=20)
                            plt.yticks(fontsize=20, fontproperties=font)

                            st.line_chart(top5_opt_performance)
                            # st.pyplot(fig)

                        # graph_cols_1 = st.columns(1)

                        # with graph_cols_1[0] :
                        #     st.write("커뮤니티 연간 시간별 에너지 프로필")
                        #     st.image('./img/EnergyMix.png')

                        

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
        st.markdown('''
                        <p class="font2"><strong> CEMOS 입력 정보 </strong></p>   
                    ''', unsafe_allow_html=True)

        st.markdown('1. 커뮤니티 위치 정보 (위도, 경도)')
        st.markdown('''
                        - 해당 커뮤니티의 위도와 경도를 값을 모를 시 구글 맵을 활용하여 위도, 경도 값 도출
                        - 현재 기본 값 설정 : 대전 유성구(위도: 36.37, 경도: 127.36)
                        - 구글 맵 Url : https://www.google.co.kr/maps
                        - 구글 맵에서 위도 및 경도 추출 방법 : https://tttsss77.tistory.com/147
                    ''')

        st.markdown('1. 커뮤니티 위치 정보* (입력필수)')
        com_loc_cols = st.columns(2)

        with com_loc_cols[0] :
            com_lat = st.text_input("위도", "", max_chars=50, placeholder='커뮤니티 위도를 입력하세요')
        with com_loc_cols[1] :
            com_lon = st.text_input("경도", "", max_chars=50, placeholder='커뮤니티 경도를 입력하세요')
        
        st.markdown("---")
        
        st.markdown('2. 커뮤니티 규모 정보-건축물 유형별 전체 연면적* (입력필수)')
        com_scale_cols = st.columns(4)

        with com_scale_cols[0] :
            com_area_off = st.text_input("업무시설 전체 연면적(m2)", "", max_chars=100, placeholder="전체 연면적을 입력해주세요")
        
        with com_scale_cols[1] :
            com_area_res = st.text_input("거주시설 전체 연면적(m2)", "", max_chars=100, placeholder="전체 연면적을 입력해주세요")

        with com_scale_cols[2] :
            com_area_etc = st.text_input("기타시설 전체 연면적(m2)", "", max_chars=100, placeholder="전체 연면적을 입력해주세요")

        with com_scale_cols[3] :
            com_occ = st.text_input("거주 총 인원(명)", "", max_chars=100, placeholder="전체 수용인원을 입력해주세요")
            
        st.markdown("---")

        st.markdown('3. 커뮤니티 연간 시간별 에너지 소비량* (입력필수)')

        uploaded_file = st.file_uploader("데이터 업로드", type = ['csv'])
        energy_df_flag = 0

        if uploaded_file is not None :
            energy_df = pd.read_csv(uploaded_file, encoding='cp949')
            # energy_df.columns = weather_cols
            energy_df_flag = 1

        st.markdown("---")
        
        st.markdown('4. 커뮤니티 현행 에너지 믹스 구성 (해당 시 작성)')

        cur_mix_cols = st.columns(6)

        with cur_mix_cols[0] :
            pv_cur = st.text_input("태양광(%)", "", max_chars=100, placeholder="0-100 사이 수를 입력하세요")
        
        with cur_mix_cols[1] :
            sr_cur = st.text_input("태양열(%)", "", max_chars=100, placeholder="0-100 사이 수를 입력하세요")

        with cur_mix_cols[2] :
            gt_cur = st.text_input("지열(%)", "", max_chars=100, placeholder="0-100 사이 수를 입력하세요")

        with cur_mix_cols[3] :
            wt_cur = st.text_input("풍력(%)", "", max_chars=100, placeholder="0-100 사이 수를 입력하세요")
        
        with cur_mix_cols[4] :
            bio_cur = st.text_input("바이오매스(%)", "", max_chars=100, placeholder="0-100 사이 수를 입력하세요")

        with cur_mix_cols[5] :
            grid_cur = st.text_input("그리드전력(%)", "", max_chars=100, placeholder="0-100 사이 수를 입력하세요")


        st.markdown("---")
        
        st.markdown('5. 선호 신재생 에너지원* (입력필수)')
        
        renew_energies = {1: '태양광', 2: '태양열', 3: '풍력', 4: '지열', 5: '수열', 6: '바이오매스', 7: '폐열',
                        8: '수소', 9: '그리드'}

        def format_func(dict, option):
                return dict[option]

        renew_info = st.multiselect('선호 신재생 에너지원을 선택하세요', options = list(renew_energies.keys()), format_func=lambda x: renew_energies[x])
        # st.write(renew_info)


        st.markdown("---")

        st.markdown('6. 설치 선호 에너지 시스템 (ESS, TSS, 연료전지, CHP, 디젤 발전)')

        energy_system = st.columns(5)
        
        with energy_system[0] :
            ess = st.text_input("ESS 용량", "", max_chars=100, placeholder="")
        
        with energy_system[1] :
            tss = st.text_input("TSS 용량", "", max_chars=100, placeholder="")
        
        with energy_system[2] :
            fuel_cell = st.text_input("연료전지 용량", "", max_chars=100, placeholder="")
        
        with energy_system[3] :
            CHP = st.text_input("열병합발전(CHP) 용량", "", max_chars=100, placeholder="")
        
        with energy_system[4] :
            diesel_gen = st.text_input("디젤 발전용량", "", max_chars=100, placeholder="")
        


        st.markdown("---")
        st.markdown('7. 에너지 믹스 평가 기준-LCC, 탄소배출량, 안전성, 안정성, 환경부하 중 하나* (입력필수)')
        
        eval_idx = {1: '비용', 2: '에너지발전효율성', 3: '탄소배출량(Co2)'}
        eval_score = st.multiselect('선호하는 평가 기준을 선택하세요', options = list(eval_idx.keys()), format_func=lambda x: eval_idx[x])

        # eval_score = st.radio("선호하는 평가 기준을 선택하세요",
        # ["LCC", "탄소배출량(Co2)", "안전성", "안정성", "환경부하"],
        # horizontal=True)
        st.write(eval_score)

        st.markdown("---")
        
        st.markdown('8. 전국 날씨 데이터 입력* (입력필수)')

        st.markdown('''
                        - 해당 커뮤니티와 가장 인접한 지역의 날씨 데이터를 입력
                            - 예: 커뮤니티 위도: 36.37, 커뮤니티 경도: 127.36 (대전) 일 경우 대전의 특정 연도에 따른 시간별 날씨 데이터 파일 업로드
                            - 기상청에서 제공하는 지역별 날씨 데이터 기준 (데이터 URL: https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36)
                                - 강원, 경기, 경남, 경북, 광주, 대구, 대전, 부산, 서울, 세종, 울산, 인천, 전남, 전북, 제주, 충남, 충북 중 택 1
                    ''')

        weather_cols = st.columns(1)

        with weather_cols[0] :
            # uploaded_file = st.file_uploader("날씨 데이터 업로드", type = ['csv'])

            def format_func(dict, option):
                return dict[option]

            weather_loc = {1: '강원', 2: '경기', 3: '경남', 4: '경북', 5: '광주', 6: '대구', 7: '대전',
                        8: '부산', 9: '서울', 10: '세종', 11: '울산', 12: '인천', 13: '전남', 14: '전북',
                        15: '제주', 16: '충남', 17: '충북'}

            weather_loc_info = st.selectbox('날씨 데이터 업로드를 위한 지역을 선택하세요', options = list(weather_loc.keys()), format_func=lambda x: weather_loc[x])
            
            
            weather_cols = ['Station', 'Region', 'Time', 'temp', 'precipitation', 'wind_speed',
                            'wind_direction', 'humidity', 'dew_temp', 'pressure', 'sunshine',
                            'solar_radiation', 'cloud', 'surface_temperature','ground_temperature']

            
            if weather_loc_info == 1 : 
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 2 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 3 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 4 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 5 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 6 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 7 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 8 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 9 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 10 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 11 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 12 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 13 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 14 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 15 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 16 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)

            elif weather_loc_info == 17 :
                weather_df = pd.read_csv("./data/weather/2020_"+str(weather_loc[weather_loc_info])+".csv", encoding='cp949')
                weather_df.columns = weather_cols
                weather_df = weather_df.fillna(0)
                # st.write(weather_df)
                

            # if uploaded_file is not None :
            #     weather_df = pd.read_csv(uploaded_file, encoding='cp949')
            #     weather_df.columns = weather_cols

        st.markdown("---")

        gen_flag = 0
                
        if com_lat == "" and com_lon == "" and com_area_off == "" and com_area_res == "" and com_area_etc == "" :
            st.error("필수 입력 정보를 입력해주세요")
    
        else :
                # if uploaded_file is not None :
            #     weather_df = pd.read_csv(uploaded_file, encoding='cp949')
            if st.button("에너지믹스 도출") :

                with st.spinner('커뮤니티 에너지믹스 계산 중'):

                    # 2020 hourly energy profile data (per net area(m2))
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

                    Building_type_1 = "Office_scale"
                    Building_type_2 = "Residential_scale"

                    # create energy profile of community on electricity and gas w.r.t total office and residential buildings
                    elec_consumption = pd.DataFrame(Energy_profile_elec[Building_type_1] * float(com_area_off) + Energy_profile_elec[Building_type_2] * float(com_area_res), columns=["Elec_consumption"])
                    gas_consumption = pd.DataFrame(Energy_profile_gas[Building_type_1] * float(com_area_off) + Energy_profile_gas[Building_type_2] * float(com_area_res), columns=["Gas_consumption"])

                    # WT: wind turbine, PV: Photovoltaic pannel -> grid ranges
                    WT_Grid = [10,100,250,750,1500]
                    PV_Grid = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300]

                    TOTAL_Price = []
                    TOTAL_SOX_amount = []
                    TOTAL_NOX_amount = []
                    TOTAL_DUST_amount = []
                    TOTAL_CO2_amount = []
                    TOTAL_PV_generation = []
                    TOTAL_WT_generation = []
                    TOTAL_Diesel_generation = []
                    TOTAL_LCC = []
                    ESS_Cap = []

                    for a in range(len(WT_Grid)) :
                        for b in range(len(PV_Grid)):
                            ONOFF = []
                            SOX_amount = []
                            NOX_amount = []
                            DUST_amount = []
                            CO2_amount = []
                            ON_Price = []
                            OFF_Price = []
                            Hourly_total_Price = []
                            Hourly_PV = []
                            Hourly_WT = []
                            Hourly_Diesel = []
                            Hourly_Diesel_Price = []

                            for i in range(len(weather_df)) :
                                OFF = elec_consumption["Elec_consumption"][i] * Hourly_Elec_Fee["Price"][i] + elec_consumption["Elec_consumption"][i] * sum(EnvironmentalFee["Price(won)"])
                                PV_generation = weather_df["solar_radiation"][i] * 277.78 * PV_Grid[b]
                                WT_generation = int(weather_df["wind_speed"][i]) * Wind_generation["{}kW".format(WT_Grid[a])][int(weather_df["wind_speed"][i])]
                                Diesel_amount = (elec_consumption["Elec_consumption"][i] - PV_generation - WT_generation)

                                if Diesel_amount < 0 :
                                    Diesel_amount = 0
                                else :
                                    Diesel_amount = Diesel_amount

                                ON = ((Diesel_amount) * 328) + ((Diesel_amount) * sum(EnvironmentalFee["Price(won)"]))

                                if OFF >= ON :
                                    ONOFF.append(0)
                                    SOX_amount.append((Diesel_amount)*(EnvironmentalFee["Amount(kg)"][0]))
                                    NOX_amount.append((Diesel_amount)*(EnvironmentalFee["Amount(kg)"][1]))
                                    DUST_amount.append((Diesel_amount)*(EnvironmentalFee["Amount(kg)"][2]))
                                    CO2_amount.append((Diesel_amount)*(EnvironmentalFee["Amount(kg)"][3]))
                                    ON_Price.append(ON)
                                    OFF_Price.append(0)

                                    if (PV_generation > elec_consumption["Elec_consumption"][i]) or (WT_generation > elec_consumption["Elec_consumption"][i]):
                                        Hourly_PV.append(elec_consumption["Elec_consumption"][i] * (PV_generation / (PV_generation + WT_generation)))
                                        Hourly_WT.append(elec_consumption["Elec_consumption"][i] * (WT_generation / (PV_generation + WT_generation)))
                                    else :
                                        Hourly_PV.append(PV_generation)
                                        Hourly_WT.append(WT_generation)

                                    Hourly_Diesel.append(Diesel_amount)
                                    Hourly_total_Price.append(ON_Price[i])

                                else :
                                    ONOFF.append(1)
                                    SOX_amount.append((elec_consumption["Elec_consumption"][i])*(EnvironmentalFee["Amount(kg)"][0]))
                                    NOX_amount.append((elec_consumption["Elec_consumption"][i])*(EnvironmentalFee["Amount(kg)"][1]))
                                    DUST_amount.append((elec_consumption["Elec_consumption"][i])*(EnvironmentalFee["Amount(kg)"][2]))
                                    CO2_amount.append((elec_consumption["Elec_consumption"][i])*(EnvironmentalFee["Amount(kg)"][3]))
                                    ON_Price.append(0)
                                    OFF_Price.append(OFF)
                                    Hourly_PV.append(0)
                                    Hourly_WT.append(0)
                                    Hourly_Diesel.append(0)
                                    Hourly_total_Price.append(OFF_Price[i])

                                i += 1

                            Annual_Price = np.sum(pd.Series(Hourly_total_Price))
                            Annual_SOX_amount = np.sum(pd.Series(SOX_amount))
                            Annual_NOX_amount = np.sum(pd.Series(NOX_amount))
                            Annual_DUST_amount = np.sum(pd.Series(DUST_amount))
                            Annual_CO2_amount = np.sum(pd.Series(CO2_amount))
                            Annual_PV_generation = np.sum(pd.Series(Hourly_PV))
                            Annual_WT_generation = np.sum(pd.Series(Hourly_WT))
                            Annual_Diesel_generation = np.sum(Hourly_Diesel)

                            # LCC공식 위치
                            Total_LCC = 0

                            for c in range (40):
                                Total_LCC += (Annual_Price * ((1.024)**(c + 1)))
                                c += 1

                            Total_LCC = Total_LCC + (PV_Grid[b] * PV_initial["Price"][0]) + (WT_Grid[a] * Wind_initial["{}kW".format(WT_Grid[a])][0])
                            ESS_Capacity = (Annual_PV_generation + Annual_WT_generation) / 365 / 0.8 / 0.9 / 0.9 * 0.7
                            Total_LCC = Total_LCC + ((ESS_Capacity * ESS_initial["Price"][0]) * 3)

                            TOTAL_Price.append(Annual_Price)
                            TOTAL_SOX_amount.append(Annual_SOX_amount)
                            TOTAL_NOX_amount.append(Annual_NOX_amount)
                            TOTAL_DUST_amount.append(Annual_DUST_amount)
                            TOTAL_CO2_amount.append(Annual_CO2_amount)
                            TOTAL_PV_generation.append(Annual_PV_generation)
                            TOTAL_WT_generation.append(Annual_WT_generation)
                            TOTAL_Diesel_generation.append(Annual_Diesel_generation)
                            TOTAL_LCC.append(Total_LCC)
                            ESS_Cap.append(ESS_Capacity)

                            b += 1
                        a += 1

                    # Final DataFrame Generation!! 
                    WT_CAP = []

                    for d in range (30):
                        WT_CAP.append(10)
                    for d in range (30):
                        WT_CAP.append(100)
                    for d in range (30):
                        WT_CAP.append(250)
                    for d in range (30):
                        WT_CAP.append(750)
                    for d in range (30):
                        WT_CAP.append(1500)

                    PV_CAP = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,
                                210,220,230,240,250,260,270,280,290,300,10,20,30,40,50,60,70,80,90,100,
                                110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,
                                290,300,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,
                                190,200,210,220,230,240,250,260,270,280,290,300,10,20,30,40,50,60,70,80,
                                90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,
                                280,290,300,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,
                                190,200,210,220,230,240,250,260,270,280,290,300]

                    st.success('Finished') 

                    Final_df = pd.DataFrame(zip(WT_CAP, PV_CAP, ESS_Cap, TOTAL_SOX_amount, TOTAL_NOX_amount, TOTAL_DUST_amount, TOTAL_PV_generation, TOTAL_WT_generation, TOTAL_Diesel_generation, TOTAL_Price, TOTAL_CO2_amount, TOTAL_LCC),
                                    columns=["풍력용량(kW)", "태양광용량(kW)", "ESS용량(kW)", "연간SOx배출량(kg)", "연간NOx배출량(kg)", "연간먼지배출량(kg)", "연간태양광발전량(kWh)", "연간풍력발전량(kWh)", "연간디젤발전량(kWh)", "연간에너지비용(원)", "연간CO2발생량(kg)", "40년간 총 LCC(원)"])

                    
                    # 오름차순으로 정렬
                    Final_Results_energy = Final_df.sort_values("연간에너지비용(원)")
                    Final_Results_lcc = Final_df.sort_values("40년간 총 LCC(원)")
                    Final_Results_co2 = Final_df.sort_values("연간CO2발생량(kg)")
                    
                    # 풍력, 태양광 용량에 따른 최적 결과 도출
                    energy_solar = Final_Results_energy.iloc[0,1]
                    energy_wind = Final_Results_energy.iloc[0,0]
                    lcc_solar = Final_Results_lcc.iloc[0,1]
                    lcc_wind = Final_Results_lcc.iloc[0,0]
                    co2_solar = Final_Results_co2.iloc[0,1]
                    co2_wind = Final_Results_co2.iloc[0,0]

                    # 각 결과 데이터프레임 (에너지 비용, Co2, LCC)
                    result_df_energy = Final_Results_energy.loc[(Final_Results_energy["태양광용량(kW)"] == energy_solar) & (Final_Results_energy["풍력용량(kW)"] == energy_wind), :]
                    result_df_lcc = Final_Results_lcc.loc[(Final_Results_lcc["태양광용량(kW)"] == lcc_solar) & (Final_Results_lcc["풍력용량(kW)"] == lcc_wind), :]
                    result_df_co2 = Final_Results_co2.loc[(Final_Results_co2["태양광용량(kW)"] == co2_solar) & (Final_Results_co2["풍력용량(kW)"] == co2_wind), :]

                    with split_cols[1] :
                        st.markdown('''
                                            <p class="font2"><strong> CEMOS 출력 정보 </strong></p>   
                                    ''', unsafe_allow_html=True)

                        table_cols = st.columns(1)

                        with table_cols[0]:               
                            st.markdown('''
                                            <p class="font2"><strong> 커뮤니티 에너지 믹스 생성 결과 통계량 </strong></p>   
                                        ''', unsafe_allow_html=True)
                            st.dataframe(Final_df.iloc[:,2:].describe())

                        # 결과들은 총 3개 그래피 및 1개 테이블로 시각화
                        # 바 차트 그래프 1,4 파이 차트 그래프 3
                        # 결과에 대한 테이블 1

                        graph_cols_1 = st.columns(1)

                        st.markdown(""" <style> .font2 {
                        font-size:20px ; font-family: 'Times, Times New Roman, Georgia, serif'; color: #000000; text-align: left;} 
                        </style> """, unsafe_allow_html=True)

                        import matplotlib.font_manager as fm

                        font_name = fm.FontProperties(fname="./font/Malgun Gothic.ttf").get_name()
                        font = fm.FontProperties(fname="./font/Malgun Gothic.ttf")
                        plt.rc('font', family=font_name)

                        plt.rcParams['font.family'] ='Malgun Gothic'
                        plt.rcParams['axes.unicode_minus'] =False


                        with graph_cols_1[0] :
                            # st.write("커뮤니티 연간 에너지 발전량")
                            st.markdown('<p class="font2"><strong>최적 에너지 믹스 구성 비율</strong></p>', unsafe_allow_html=True)
                            
                            temp = result_df_energy.iloc[0,6:9].values
                            total_sum = temp.sum()
                            exp = [0, 0.4, 0.5]
                            fig = plt.figure(figsize=(10,9))
                            plt.rc('font', family = 'Malgun Gothic' )
                            # plt.pie((temp/total_sum)*100, labels = result_df_energy.iloc[:,6:9].columns,
                            #          autopct='%.2f%%', explode=exp, rotatelabels=True)

                            plt.pie((temp/total_sum)*100, pctdistance=1.15, autopct='%.2f%%', explode = [0.2, 0.2, 0], 
                                        textprops={'fontsize': 15})
                            plt.legend(loc='upper right', labels = result_df_energy.iloc[:,6:9].columns, fontsize=20, prop=font)

                            st.pyplot(fig)

                        graph_cols_2 = st.columns(1)
                        with graph_cols_2[0] :
                            st.markdown('<p class="font2"><strong>최적 에너지 믹스 성능 평가</strong></p>', unsafe_allow_html=True)
                            # st.write("에너지믹스 비용 산정: 연간에너지 비용, 연간 CO2 발생량, 40년간 총 LCC")
                            # temp = result_df_energy.iloc[0, 9:].values

                            # table_1_df = pd.DataFrame([temp], columns=result_df_energy.iloc[:, 9:].columns)
                            # st.table(table_1_df.T)
                            temp = Final_Results_lcc.iloc[0,[9,11]].values

                            bar_graph_2_df = pd.DataFrame([temp])
                            bar_graph_2_df.columns = ["연간에너지비용(원)", "40년간 총 LCC(원)"]

                            fig = plt.figure(figsize=(13,6))
                            ax = fig.add_subplot(111)

                            rects = plt.barh(bar_graph_2_df.columns, bar_graph_2_df.iloc[0,:], color=['r','g','b'], align='center', height=0.5)
                            # plt.yticks(ypos, industry)

                            for i, rect in enumerate(rects):
                                ax.text(1 * rect.get_width(), rect.get_y() + rect.get_height() / 2.0, round(bar_graph_2_df.iloc[0,:][i], 2), fontsize=13)

                            plt.xticks(fontsize=20)
                            plt.yticks(fontsize=20, fontproperties=font)

                            st.pyplot(fig)

                        
                        # graph_cols_1 = st.columns(1)

                        # with graph_cols_1[0] :
                        #     st.write("커뮤니티 연간 시간별 에너지 프로필")
                        #     st.image('./img/EnergyMix.png')

                        graph_cols_3 = st.columns(1)

                        with graph_cols_3[0] :
                            st.markdown('<p class="font2"><strong>현행 및 최적 에너지믹스 비교 평가</strong></p>', unsafe_allow_html=True)
                            
                            final_hourly_profile = pd.DataFrame(zip(ONOFF, Hourly_PV, Hourly_WT, Hourly_Diesel))
                            total_elec = round(elec_consumption["Elec_consumption"], 2)

                            final_hourly_profile = pd.concat([total_elec, final_hourly_profile], axis=1)

                            final_hourly_profile.columns = ['Total_Elec','ONOFF','PV','WT','Diesel']
                            on_site_energy = final_hourly_profile.iloc[:,2]+final_hourly_profile.iloc[:,3]+final_hourly_profile.iloc[:,4]
                            off_site_energy = final_hourly_profile.iloc[:,0] - on_site_energy

                            idx = np.arange(off_site_energy.shape[0])

                            fig = plt.figure(figsize=(12,6))
                            # plt.rc('font', family = 'Malgun Gothic' )
                            plt.plot(idx, on_site_energy , label='커뮤니티 내부 에너지 발전량')
                            plt.plot(idx, final_hourly_profile.iloc[:,0], label='커뮤니티 외부 에너지 필요량')

                            plt.legend(fontsize=65, prop=font)

                            plt.xlabel('연간 시간 ',  fontproperties=font)
                            plt.ylabel('에너지 (kWh)', fontproperties=font)
                            # plt.xlabel('연간 시간 ', fontsize=60, fontproperties=font)
                            # plt.ylabel('에너지 (kWh)', fontsize=60, fontproperties=font)
                            # # plt.xticks(fontsize=10)
                            # plt.yticks(fontsize=10)
                            plt.ylim(0, 4000)

                            st.pyplot(fig)
                            

                        graph_cols_4 = st.columns(1)

                        with graph_cols_4[0] :
                            st.markdown('<p class="font2"><strong>연간 유해물질 배출량</strong></p>', unsafe_allow_html=True)
                            
                            temp = result_df_energy.iloc[0,[3,4,5,10]].values

                            bar_graph_3_df = pd.DataFrame([temp])
                            bar_graph_3_df.columns = ["연간SOx배출량(kg)", "연간NOx배출량(kg)", "연간먼지배출량(kg)", "연간CO2발생량(kg)"]


                            # fig = plt.figure(figsize=(17, 10))
                            fig = plt.figure(figsize=(13,9))
                            ax = fig.add_subplot(111)

                            rects = plt.barh(bar_graph_3_df.columns, bar_graph_3_df.iloc[0,:], color=['r','g','b','y'], align='center', height=0.5)
                            # plt.yticks(ypos, industry)

                            for i, rect in enumerate(rects):
                                ax.text(1 * rect.get_width(), rect.get_y() + rect.get_height() / 2.0, round(bar_graph_3_df.iloc[0,:][i], 2), fontsize=13)

                            plt.xticks(fontsize=20)
                            plt.yticks(fontsize=20, fontproperties=font)

                            st.pyplot(fig)
                            


                        graph_cols_5 = st.columns(1)

                        with graph_cols_5[0] :
                            # st.write("커뮤니티 연간 유해물질 배출량")
                            st.markdown('<p class="font2"><strong>요약 리포트</strong></p>', unsafe_allow_html=True)

                            plt.rc('font', family = 'Malgun Gothic' )

                            bar_graph_1_df= result_df_energy.iloc[:,:3]
                            bar_graph_1_df['ESS용량(kW)'] = bar_graph_1_df['ESS용량(kW)']/1000
                            bar_graph_1_df.columns = ["풍력용량(kW)", "태양광용량(kW)", "ESS용량(1000kw)"]

                            # graph_1_df.iloc[0,:].head()
                            # graph_1_df.iloc[0,:]

                            fig = plt.figure(figsize=(10,6))
                            plt.barh(bar_graph_1_df.columns, bar_graph_1_df.iloc[0,:], color=['r','g','b'], align='center', height=0.5)
                            
                            plt.xticks(fontsize=20)
                            plt.yticks(fontsize=20, fontproperties=font)

                            st.pyplot(fig)



    
page_names_to_funcs = {
    "홈": home,
    "커뮤니티 에너지 믹스 데이터 현황": data_summary,
    "커뮤니티 에너지 믹스 탐색(신규)": cemos_new,
    "커뮤니티 에너지 믹스 탐색(기존)": cemos
    }

page_names_to_funcs[selected]()





