#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # Competition을 위한 데이터 처리 예제

# - mem_data.csv 파일의 GENDER 열과 MEM_ID열은 모델링에 사용됨에 따라 수정하시면 안됩니다.

# In[2]:


mem_data = pd.read_csv('mem_data.csv')
mem_tr = pd.read_csv('mem_transaction.csv')
s_info = pd.read_csv('store_info.csv')


# In[3]:


mem_data.info()


# In[4]:


mem_tr.head()


# In[5]:


s_info.head()


# In[6]:


mem_data.GENDER.value_counts()


# ## **[mem_data 처리]**

# **[변수 처리 1]** SMS 수신동의 정수 처리하기

# In[7]:


mem_data.SMS.value_counts()


# In[8]:


mem_data.SMS = (mem_data.SMS=='Y').astype(int)
mem_data.SMS.value_counts()


# **[변수 처리 2]** 양/음력(BIRTH_SL) 정수 처리하기 

# In[9]:


mem_data.BIRTH_SL = (mem_data.BIRTH_SL=='S').astype(int)
mem_data.BIRTH_SL.value_counts()


# In[10]:


mem_data.head()


# **[변수 처리 3]** 구매 합계(SALES_AMT) 로그 처리 하기
# - 참고 : 로그처리에는 음수가 들어갈 수 없음 / 0이 있을 경우 1을 더하고 처리

# In[11]:


f = mem_data.SALES_AMT.where(mem_data.SALES_AMT>=0, other=0) # 음수처리
f = np.log(f+1)
mem_data.SALES_AMT = f


# **[변수 처리 4]** 로그 처리가 필요하다고 생각되는 변수를 로그처리 하기

# In[12]:


f = mem_data.ACC_PNT.where(mem_data.ACC_PNT>=0, other=0) # 음수처리
f = np.log(f+1)
mem_data.ACC_PNT = f


# **[변수 생성 1]** 최근 방문 일자(LAST_VST_DT)로부터 경과일 구하기

# In[13]:


f = pd.to_datetime(mem_data.LAST_VST_DT) # 날짜가 Object 이므로 숫자 처리
f = (pd.to_datetime('2007-12-13') - f).dt.days #방문이 얼마나 오래되었는지 ,time 등으로 알고 싶은 것으로 바꿀 수 있다.
mem_data['E_DAY'] = f
mem_data.E_DAY.head()


# In[14]:


f = pd.to_datetime(mem_data.LAST_VST_DT)


# In[15]:


f = (pd.to_datetime('2007-12-13') - f)


# In[16]:


f[f.isna()]


# In[17]:


mem_data.LAST_VST_DT[1612]


# In[18]:


f.isna().sum()


# **[변수 생성 2]** 등록일(RGST_DT)로부터 경과일 구하기

# In[19]:


f = pd.to_datetime(mem_data.RGST_DT) # 날짜가 Object 이므로 숫자 처리
f = (pd.to_datetime('2007-12-13') - f).dt.days #방문이 얼마나 오래되었는지 ,time 등으로 알고 싶은 것으로 바꿀 수 있다.
mem_data['E_DAY'] = f
mem_data.E_DAY.head()


# **[변수 생성 3]** 우편번호(ZIP_CD)에서 광역행정구역 데이터 가져오기
# - 참고: 6자리 우편번호의 구성 : https://ko.wikipedia.org/wiki/대한민국의_우편번호#6자리_우편번호_(1988~2015)

# In[20]:


mem_data.ZIP_CD.head()


# In[21]:


mem_data.ZIP_CD[1][0]


# In[22]:


# 리스트를 구할때 첫번째 글자만 가져올때
f = [x[0] for x in mem_data.ZIP_CD]


# In[23]:


f = [x[0] for x in mem_data.ZIP_CD]
mem_data['R_REGION'] = f
mem_data.R_REGION = mem_data.R_REGION.where(mem_data.R_REGION != '-', other=0).astype(int)
mem_data.R_REGION.head()


# **[변수 생성 4]** 우편번호(ZIP_CD)에서 광역행정구역과 구를 합쳐서 데이터 가져오기

# In[24]:


f=mem_data.groupby('ZIP_CD')['R_REGION'].agg({'sum'}).reset_index()
mem_data=mem_data.merge(f,how='left') 
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0)


# ## **[다른 데이터와 연동하여 변수 생성]**

# **[변수 생성 1]** 평균 구매액 구하기

# In[25]:


f = mem_tr.groupby('MEM_ID')['SELL_AMT'].agg({'mean'}).reset_index()
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0)


# **[변수 생성 2]** 포인트 적립 횟수 구하기

# In[26]:


f = mem_tr[mem_tr.MEMP_TP=='A'].groupby('MEM_ID')['SELL_AMT'].agg({'size'}).reset_index()
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0)


# **[변수 생성 3]** 요일 구매 패턴 구하기 : 주중형 / 주말형

# In[27]:


def weekday(x):
    w = x.dayofweek 
    if w < 4:
        return 1 # 주중
    else:
        return 0 # 주말
f = mem_tr.groupby('MEM_ID')['SELL_DT'].agg([('요일구매패턴', lambda x : pd.to_datetime(x).apply(weekday).value_counts().index[0])]).reset_index()
mem_data = mem_data.merge(f, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0)


# **[변수 생성 4]** 시간대별 표인트 적립 건수 구하기 : Morning(09-12) / Afternoon(13-17) / Evening(18-20)

# In[28]:


def f1(x):
    k = x.month
    if 9 <= k <= 12 :
        return('MORNING')
    elif 13 <= k <= 17 :
        return('AFTERNOON')
    else :
        return('EVENING')    
    
mem_tr['MEMP_DT'] = pd.to_datetime(mem_tr.SELL_DT).apply(f1)
f = pd.pivot_table(mem_tr, index='MEM_ID', columns='MEMP_DT', values='MEMP_TP', 
                   aggfunc=np.size, fill_value=0).reset_index()


# ## **[최종 결과 저장]**

# - mem_data에 최종 결과가 모일 수 있도록 준비
# - 1.예측에 사용하지 않을 열들 drop
# - 2.csv형태로 저장
# - **주의사항1 : MEM_ID열은 삭제하지 마세요. **
# - MEM_ID를 변수로 사용하기를 원하면 다른 열을 새로 생성해서 복사해서 사용하세요.
# - 주의사항2 : GENDER열을 제외하고 문자열이 포함된 열이 데이터에 포함되지 않도록 한다.

# In[29]:


mem_data.info()


# In[30]:


d_col = ['BIRTH_SL','M_STORE_ID','BIRTH_DT','ZIP_CD','RGST_DT','LAST_VST_DT']
mem_data = mem_data.drop(d_col, axis=1)
mem_data.info()


# In[31]:


mem_data.head()


# In[32]:


final_data = pd.read_csv('final_data.csv')


# In[33]:


final_data.dtypes


# In[34]:


mem_data = mem_data.merge(mem_tr, how='left')
mem_data.iloc[:,-1] = mem_data.iloc[:,-1].fillna(0)


# In[35]:


mem_data.head()


# In[36]:


mem_data.to_csv('final_data.csv', index=False)


# In[37]:


final_data = pd.read_csv('final_data.csv')


# In[38]:


final_data.dtypes


# In[41]:


d_col = ['MEMP_STY', 'MEMP_DT', 'MEMP_TP']
x = final_data.drop(d_col, axis=1)
x.info()
print(x.shape)


# In[40]:


np.random.seed(123)

tr = x.GENDER!='UNKNOWN'
train = x[tr]
train.GENDER = (train.GENDER=='M').astype(int)
train.shape


# In[43]:


te = x.GENDER=='UNKNOWN'
test = x[te].sort_values('MEM_ID')
test.head(3)
print(test.shape)


# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import glob


# In[45]:


kfold = StratifiedKFold(n_splits=5) # 하이퍼 파라미터 지정
n_it = 12


# In[46]:


t_final = test[['MEM_ID', 'GENDER']]
t_final.head()


# In[47]:


t_final.shape


# In[48]:


test = test.drop(['GENDER','MEM_ID'], axis=1)
target = train.GENDER.values
train = train.drop(['GENDER','MEM_ID'], axis=1)


# In[49]:


print(train.shape, test.shape)


# In[50]:


params = {'max_features':list(np.arange(1, train.shape[1])), 'bootstrap':[False], 'n_estimators': [50], 'criterion':['gini','entropy']}
model = RandomizedSearchCV(RandomForestClassifier(), param_distributions=params, n_iter=n_it, cv=kfold, scoring='roc_auc',n_jobs=-1, verbose=1)
print('MODELING.............................................................................')
model.fit(train, target)
print('========BEST_AUC_SCORE = ', model.best_score_)
model = model.best_estimator_
t_final.GENDER = model.predict_proba(test.values)[:,1]


# In[ ]:


train.info()


# In[ ]:




