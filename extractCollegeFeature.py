# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn import preprocessing
import math

############################################
###############数据提取######################
#基本信息栏，共28项，部分信息冗余，待优化
base_info_columns = ['id','time_submit','time_spend','source','source_detail','IP','sex','date_born','college','date_gaokao','grade_gaokao',
           'rank_province','high_school','area_high_school','type_high_school','subject_advicer','rank_advicer',
           'grade_exam1','rank_exam1','grade_exam2','rank_exam2','grade_exam3','rank_exam3','amount_grade3',
            'type_grade_change','date_zhongkao','grade_zhongkao','rank_zhongkao']
#问卷栏，从问题15到问题323 + 邮箱题目
whole_question_columns = list(range(15,325))
college_data = pd.read_csv('data/sourceData/college_data.csv',header=0,dtype=str,encoding='gbk')
college_data.columns = base_info_columns + whole_question_columns
college_data = college_data[college_data['grade_zhongkao']!='0']
college_data = college_data[college_data['grade_gaokao']!='0']
college_data = college_data[college_data['rank_province']!='0']
# print(college_data.info())


###############数据清洗#######################
##########客观基础能力#########
"""
    1.一练成绩
    2.一练排名
    3.二练成绩
    4.二练排名
    5.三练成绩
    6.三练排名
    7.中考成绩
    8.中考排名
"""
base_info = college_data.iloc[:,0:28].copy()
BI = base_info[['id','grade_exam1','rank_exam1','grade_exam2','rank_exam2','grade_exam3',
                'rank_exam3','grade_zhongkao','rank_zhongkao']].copy()
BI.replace('(空)',0,inplace=True)
BI = BI.astype('int')
train_data_college = BI.copy()
# print(train_data_college.info())
# print(train_data_college[:5])


##########问卷整理部分#################
###########个人影响组################
questionnaire_info = college_data.iloc[:,28:-2]
"""
    2.1. 自我效能感（Self Efficacy）
    - 问卷位置： 15-24
    - 选项计分方式：4-3-2-1
    - 共10题
    - 无分量表
"""
SE = questionnaire_info.iloc[:,:10].copy()
SE.replace("1",4,inplace=True)
SE.replace("2",3,inplace=True)
SE.replace("3",2,inplace=True)
SE.replace("4",1,inplace=True)
SE['SE_points'] = SE.apply(lambda x: x.sum(),axis=1)
train_data_college["SE_points"] = SE.loc[:,['SE_points']]
# print(SE.info())
# print(train_data_college)

"""
    2.2. 成就动机（Achievement Motivation）
    - 问卷位置： 25-54
    - 选项计分方式：4-3-2-1
    - 共30题
    - 成功动机分量表：MS(motivation-of-success)、1-15题
    - 失败动机分量表：MF(motivation-of-failure)、16-30题
"""
AM = questionnaire_info.iloc[:,10:40].copy()
AM.columns=list(range(1,31))
AM.replace("1",4,inplace=True)
AM.replace("2",3,inplace=True)
AM.replace("3",2,inplace=True)
AM.replace("4",1,inplace=True)
AM_MS = AM.loc[:,0:15].copy()
AM_MF = AM.loc[:,15:30].copy()
AM_MS['AM_MS_points'] = AM_MS.apply(lambda x: x.sum(),axis=1)
AM_MF['AM_MF_points'] = AM_MF.apply(lambda x: x.sum(),axis=1)
train_data_college['AM_MS_points'] = AM_MS.loc[:,['AM_MS_points']]
train_data_college['AM_MF_points'] = AM_MF.loc[:,['AM_MF_points']]
# print(AM.info())
# print(train_data_college)

"""
    2.3. 学习策略（Learning Strategy）
    - 问卷位置： 55-99
    - 选项计分方式：4-3-2-1
    - 反向计分题号：2,6,15,18,22,26,30,36,38,40
    - 共45题
    - 认知策略分量表：CS(cognitive strategy)、题号：4,6,9,12,19,22,26,28,32,34,37,39,42,44,45
    - 元认知分策略量表：MS(metacognitive strategy)、题号：1,3,7,14,15,17,18,20,21,24,27,29,36,41
    - 资源管理利用策略分量表：RM(Resource management)、题号：2,5,8,10,11,13,16,23,25,30,31,33,35,38,40,43
"""
LS = questionnaire_info.iloc[:,40:85].copy()
LS = LS.astype('int')
LS.columns = list(range(1,46))
LS_reverse = LS[[2,6,15,18,22,26,30,36,38,40]].copy()
LS.drop([2,6,15,18,22,26,30,36,38,40],axis=1,inplace=True)
LS_reverse.replace("1",5,inplace=True)
LS_reverse.replace("2",4,inplace=True)
LS_reverse.replace("3",3,inplace=True)
LS_reverse.replace("4",2,inplace=True)
LS_reverse.replace("5",1,inplace=True)
LS = pd.concat([LS,LS_reverse],axis=1)
LS_CS = LS[[4,6,9,12,19,22,26,28,32,34,37,39,42,44,45]].copy()
LS_MS = LS[[1,3,7,14,15,17,18,20,21,24,27,29,36,41]].copy()
LS_RM = LS[[2,5,8,10,11,13,16,23,25,30,31,33,35,38,40,43]].copy()
LS_CS['LS_CS_points'] = LS_CS.apply(lambda x: x.sum(),axis=1)
LS_MS['LS_MS_points'] = LS_MS.apply(lambda x: x.sum(),axis=1)
LS_RM['LS_RM_points'] = LS_RM.apply(lambda x: x.sum(),axis=1)
train_data_college['LS_CS_points'] = LS_CS.loc[:,['LS_CS_points']]
train_data_college['LS_MS_points'] = LS_MS.loc[:,['LS_MS_points']]
train_data_college['LS_RM_points'] = LS_RM.loc[:,['LS_RM_points']]
# print(LS.info())
# print(train_data_college)


"""
    2.4. 考试焦虑（Examination Anxiety）
    - 问卷位置： 100-136
    - 选项计分方式：1-0
    - 反向计分题号：3,15,26,27,29,33
    - 共37题
    - 无分量表
"""
EA = questionnaire_info.iloc[:,85:122].copy()
EA = EA.astype('int')
EA.columns = list(range(1,38))
EA_reverse = EA[[3,15,26,27,29,33]].copy()
EA.drop([3,15,26,27,29,33],axis=1,inplace=True)
EA_reverse.replace(1,0,inplace=True)
EA_reverse.replace(2,1,inplace=True)
EA.replace(2,0,inplace=True)
EA = pd.concat([EA,EA_reverse],axis=1)
EA['EA_points'] = EA.apply(lambda x:x.sum(),axis=1)
train_data_college['EA_points'] = EA.loc[:,['EA_points']]
# print(EA.info())
# print(train_data_college)

####################

"""
    3.1.家庭环境(Home Environment)
    - 问卷位置：137-226
    - 选项计分方式：1-2
    - 计正分题号：11,41,61,2,22,52,72,13,33,63,4,54,55,65,16,36,46,76,7,27,57,87,18,38,88,9,29,49,79,10,20,60,70
    - 共90道题
    - 10个分量表
    - 亲密度：11,41,61,1,21,31,51,71,81
    - 情感表达：2,22,52,72,12,32,42,62,82
    - 矛盾性：13,33,63,3,23,43,53,73,83
    - 独立性：4,54,14,24,34,44,64,74,84
    - 成功性：55,65m,5,15,25,35,45,75,85
    - 知识性：16,36,46,76,6,26,56,66,86
    - 娱乐性：7,27,57,87,17,37,47,67,77
    - 道德宗教观：18,38,88,8,28,48,58,68,78
    - 组织性：9,29,49,79,19,39,59,69,89
    - 控制性：10,20,60,70,30,40,50,80,90
"""
HE = questionnaire_info.iloc[:,122:212].copy()
HE = HE.astype('int')
HE.columns = list(range(1,91))
HE_normal = HE[[11,41,61,2,22,52,72,13,33,63,4,54,55,65,16,36,46,76,7,27,57,87,18,38,88,9,29,49,79,10,20,60,70]].copy()
HE.drop([11,41,61,2,22,52,72,13,33,63,4,54,55,65,16,36,46,76,7,27,57,87,18,38,88,9,29,49,79,10,20,60,70],axis=1,inplace=True)
HE.replace(1,-1,inplace=True)
HE.replace(2,-2,inplace=True)
HE = pd.concat([HE,HE_normal],axis=1)
HE_QM = HE[[11,41,61,1,21,31,51,71,81]].copy()
HE_BD = HE[[2,22,52,72,12,32,42,62,82]].copy()
HE_MD = HE[[13,33,63,3,23,43,53,73,83]].copy()
HE_DL = HE[[4,54,14,24,34,44,64,74,84]].copy()
HE_CG = HE[[55,65,5,15,25,35,45,75,85]].copy()
HE_ZS = HE[[16,36,46,76,6,26,56,66,86]].copy()
HE_YL = HE[[7,27,57,87,17,37,47,67,77]].copy()
HE_ZJ = HE[[18,38,88,8,28,48,58,68,78]].copy()
HE_ZZ = HE[[9,29,49,79,19,39,59,69,89]].copy()
HE_KZ = HE[[10,20,60,70,30,40,50,80,90]].copy()
HE_QM['HE_QM_points'] = HE_QM.apply(lambda x:x.sum(),axis=1)
HE_BD['HE_BD_points'] = HE_BD.apply(lambda x:x.sum(),axis=1)
HE_MD['HE_MD_points'] = HE_MD.apply(lambda x:x.sum(),axis=1)
HE_DL['HE_DL_points'] = HE_DL.apply(lambda x:x.sum(),axis=1)
HE_CG['HE_CG_points'] = HE_CG.apply(lambda x:x.sum(),axis=1)
HE_ZS['HE_ZS_points'] = HE_ZS.apply(lambda x:x.sum(),axis=1)
HE_YL['HE_YL_points'] = HE_YL.apply(lambda x:x.sum(),axis=1)
HE_ZJ['HE_ZJ_points'] = HE_ZJ.apply(lambda x:x.sum(),axis=1)
HE_ZZ['HE_ZZ_points'] = HE_ZZ.apply(lambda x:x.sum(),axis=1)
HE_KZ['HE_KZ_points'] = HE_KZ.apply(lambda x:x.sum(),axis=1)
train_data_college['HE_QM_points'] = HE_QM.loc[:,['HE_QM_points']]
train_data_college['HE_BD_points'] = HE_BD.loc[:,['HE_BD_points']]
train_data_college['HE_MD_points'] = HE_MD.loc[:,['HE_MD_points']]
train_data_college['HE_DL_points'] = HE_DL.loc[:,['HE_DL_points']]
train_data_college['HE_CG_points'] = HE_CG.loc[:,['HE_CG_points']]
train_data_college['HE_ZS_points'] = HE_ZS.loc[:,['HE_ZS_points']]
train_data_college['HE_YL_points'] = HE_YL.loc[:,['HE_YL_points']]
train_data_college['HE_ZJ_points'] = HE_ZJ.loc[:,['HE_ZJ_points']]
train_data_college['HE_ZZ_points'] = HE_ZZ.loc[:,['HE_ZZ_points']]
train_data_college['HE_KZ_points'] = HE_KZ.loc[:,['HE_KZ_points']]
# print(HE.info())
# print(train_data_college)


"""
    3.2.家庭教养方式(Parenting Pattern)
    - 问卷位置：227-292
    - 问卷计分方式：1-2-3-4
    - 共66题
    - 共分量表
    - 温暖理解：2,4,6,7,9,15,20,25,29,30,31,32,33,37,42,54,60,61,66,44,63
    - 惩罚严厉：13,17,43,51,52,53,55,58,62,5,18,49
    - 过分干涉：1,10,11,14,27,36,48,50,56,57,12,16,19,24,35,41,59
    - 偏爱被试：3,8,22,64,65,32
    - 拒绝否认：21,23,28,34,35,45,26,38,39,47
    - 干涉保护：1,10,11,14,27,36,48,50,56,57,12,16,19,24,35,41,59
"""
PP = questionnaire_info.iloc[:,212:278].copy()
PP = PP.astype('int')
PP.columns = list(range(1,67))
PP_WN = PP[[2,4,6,7,9,15,20,25,29,30,31,32,33,37,42,54,60,61,66,44,63]].copy()
PP_CF = PP[[13,17,43,51,52,53,55,58,62,5,18,49]].copy()
PP_GF = PP[[1,10,11,14,27,36,48,50,56,57,12,16,19,24,35,41,59]].copy()
PP_PA = PP[[3,8,22,64,65,32]].copy()
PP_JJ = PP[[21,23,28,34,35,45,26,38,39,47]].copy()
PP_GS = PP[[1,10,11,14,27,36,48,50,56,57,12,16,19,24,35,41,59]].copy()
PP_WN['PP_WN_points'] = PP_WN.apply(lambda x:x.sum(),axis=1)
PP_CF['PP_CF_points'] = PP_CF.apply(lambda x:x.sum(),axis=1)
PP_GF['PP_GF_points'] = PP_GF.apply(lambda x:x.sum(),axis=1)
PP_PA['PP_PA_points'] = PP_PA.apply(lambda x:x.sum(),axis=1)
PP_JJ['PP_JJ_points'] = PP_JJ.apply(lambda x:x.sum(),axis=1)
PP_GS['PP_GS_points'] = PP_GS.apply(lambda x:x.sum(),axis=1)
train_data_college['PP_WN_points'] = PP_WN.loc[:,['PP_WN_points']]
train_data_college['PP_CF_points'] = PP_CF.loc[:,['PP_CF_points']]
train_data_college['PP_GF_points'] = PP_GF.loc[:,['PP_GF_points']]
train_data_college['PP_PA_points'] = PP_PA.loc[:,['PP_PA_points']]
train_data_college['PP_JJ_points'] = PP_JJ.loc[:,['PP_JJ_points']]
train_data_college['PP_GS_points'] = PP_GS.loc[:,['PP_GS_points']]
# print(PP[:5])
# print(train_data_college.info())


"""
    3.3.家庭亲密性和适应性（Familial Intimacy and Adaptability.）
    - 问卷位置： 293-323
    - 选项计分方式：1-2-3-4-5
    - 计负分题号：3,9,19,29,24,28
    - 共30题
    - 亲密度分量表：初始分36 + 1,5,7,11,13,15,17,21,23,25,27,30,3,9,19,29
    - 适应性分量表：初始分12 + 2,4,6,8,10,12,14,16,18,20,22,26,24,28
"""
FIA = questionnaire_info.iloc[:,278:].copy()
FIA = FIA.astype('int')
FIA.columns = list(range(1,31))
FIA_reverse = FIA[[3,9,19,29,24,28]].copy()
FIA.drop([3,9,19,29,24,28],axis=1,inplace=True)
FIA_reverse.replace(1,-1,inplace=True)
FIA_reverse.replace(2,-2,inplace=True)
FIA_reverse.replace(3,-3,inplace=True)
FIA_reverse.replace(4,-4,inplace=True)
FIA_reverse.replace(5,-5,inplace=True)
FIA = pd.concat([FIA,FIA_reverse],axis=1)
FIA_I = FIA[[1,5,7,11,13,15,17,21,23,25,27,30,3,9,19,29]].copy()
FIA_A = FIA[[2,4,6,8,10,12,14,16,18,20,22,26,24,28]].copy()
FIA_I['FIA_I_points'] = FIA_I.apply(lambda x:x.sum(),axis=1)
FIA_I['FIA_I_points'] = FIA_I['FIA_I_points'] + 36
FIA_A['FIA_A_points'] = FIA_A.apply(lambda x:x.sum(),axis=1)
FIA_A['FIA_A_points'] = FIA_A['FIA_A_points'] + 12
train_data_college['FIA_I_points'] = FIA_I.loc[:,['FIA_I_points']]
train_data_college['FIA_A_points'] = FIA_A.loc[:,['FIA_A_points']]
# print(FIA.info())
# print(train_data_college.info())
# print(train_data_college[:5])

##################生成标签#####################
def extract_province(s):
    if('上海' in s):
        return 660
    elif('江苏' in s):
        return 480
    else:
        return 750
tmp = college_data[['id','grade_gaokao','rank_province','IP']].copy()
tmp['top_gaokao'] = tmp['IP'].apply(extract_province)
tmp['grade_gaokao'] = tmp['grade_gaokao'].astype('int')
tmp['rank_province'] = tmp['rank_province'].astype('int')
tmp = tmp[tmp['rank_province']<50000]       #取省前5万名
tmp['grade_gaokao1'] = tmp['grade_gaokao']/tmp['top_gaokao']        #分数除以省最高分求比例
tmp = tmp[tmp['grade_gaokao1']<1]           #所填分数大于省最高分时视为无效
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
tmp['rank_province1'] = min_max_scaler.fit_transform(tmp['rank_province'].values.reshape(-1,1))
tmp['rank_province1'] = 1 - tmp['rank_province1']
tmp = tmp.sort_values(['rank_province1'],ascending=False)
tmp['rank_province2'] = list(range(1,tmp['rank_province1'].count()+1))
tmp = tmp.sort_values(['grade_gaokao1'],ascending=False)
tmp['grade_gaokao2'] = list(range(1,tmp['grade_gaokao1'].count()+1))
tmp['distance'] = abs(tmp['grade_gaokao2'] - tmp['rank_province2'])
tmp['distance'] = min_max_scaler.fit_transform(tmp['distance'].values.reshape(-1,1))
tmp['distance1'] = tmp['distance'].apply(lambda x:math.exp(x))          #排名和成绩差距作为惩罚
tmp['distance1'] = min_max_scaler.fit_transform(tmp['distance1'].values.reshape(-1,1))
a,b = 1,0        #将归一化后的高考成绩和排名加权   加权系数 a  b
k = 0.1        #减去排名不一致的惩罚系数 K
tmp['sum_gaokao'] = a * tmp['grade_gaokao1']  + b * tmp['rank_province1']
tmp['final_grade'] = tmp['sum_gaokao'] - k*tmp['distance1']
tmp['final_grade'] = min_max_scaler.fit_transform(tmp['final_grade'].values.reshape(-1,1))
tmp2 = tmp[['id','final_grade']].copy()
tmp2['id'] = tmp2['id'].astype('int')
# print(tmp[['grade_gaokao','grade_gaokao1','rank_province','rank_province1','final_grade']])
train_data_college['id'] = train_data_college['id'].astype('int')
train_data_college = pd.merge(train_data_college,tmp2,how = 'inner',on='id')

# print(train_data_college.info())
train_data_college.to_csv('data/midData/train_data_college.csv',index=None)