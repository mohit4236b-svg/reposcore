import os
try:
    from openai import OpenAI
    O=1
except:O=0
try:
    from google import genai
    from google.genai import types
    G=1
except:G=0
P="Review GitHub repo from README and metrics. Be specific, critical, no hype. Cover: 1) README purpose/features/tech/arch. 2) How metrics (stars, activity) align with README claims. 3) 1-2 actionable gaps. 4) Strengths in README/setup. 3-5 sentences. No model/prediction/confidence.\nRepo: {full_name}\n★:{stars} | Fork:{forks} | Issues:{open_issues}\nAge:{repo_age_days}d | Since commit:{last_commit_days}d\nContributors:{total_contributors}\nTopics:{topics}\nLang:{primary_language}\nCI:{has_ci} | Tests:{has_tests} | Lic:{has_license}\nREADME len:{readme_size}\nREADME:\n{readme_for_prompt}"
def N(r,f,p,pr):
    if not O:return{"review":"AI review unavailable: openai not installed.","status":"skipped","provider":"nvidia"}
    k=os.getenv("NVIDIA_API_KEY")
    if not k or k.strip()=="":return{"review":"AI review unavailable: NVIDIA_API_KEY not set.","status":"skipped","provider":"nvidia"}
    m=8000
    rp=r[:m]
    if len(r)>m:rp+="\n\n[README truncated from "+str(len(r))+" to "+str(m)+" chars]"
    pt=P.format(full_name=f.get("full_name","Unknown"),stars=f.get("stars",0),forks=f.get("forks",0),open_issues=f.get("open_issues",0),repo_age_days=f.get("repo_age_days",0),last_commit_days=f.get("last_commit_days",0),total_contributors=f.get("total_contributors",0),topics=", ".join(f.get("topics",[])) if f.get("topics") else "None",primary_language=f.get("primary_language","Unknown"),has_ci="Yes" if f.get("has_ci") else "No",has_tests="Yes" if f.get("has_tests") else "No",has_license="Yes" if f.get("has_license") else "No",readme_size=f.get("readme_size",0),readme_for_prompt=rp)
    try:
        c=OpenAI(base_url="https://integrate.api.nvidia.com/v1",api_key=k)
        resp=c.chat.completions.create(model="meta/llama-3.1-70b-instruct",messages=[{"role":"user","content":pt}],temperature=0.3,max_tokens=500)
        if resp.choices[0].message.content:return{"review":resp.choices[0].message.content.strip(),"status":"success","provider":"nvidia"}
        else:return{"review":"AI review unavailable: Empty response.","status":"error","provider":"nvidia"}
    except Exception as e:return{"review":"AI review unavailable: "+str(e),"status":"error","provider":"nvidia"}
def g_(r,f,p,pr):
    if not G:return{"review":"AI review unavailable: google-genai not installed.","status":"skipped","provider":"gemini"}
    k=os.getenv("GEMINI_API_KEY")
    if not k or k.strip()=="":return{"review":"AI review unavailable: GEMINI_API_KEY not set.","status":"skipped","provider":"gemini"}
    m=8000
    rp=r[:m]
    if len(r)>m:rp+="\n\n[README truncated from "+str(len(r))+" to "+str(m)+" chars]"
    pt=P.format(full_name=f.get("full_name","Unknown"),stars=f.get("stars",0),forks=f.get("forks",0),open_issues=f.get("open_issues",0),repo_age_days=f.get("repo_age_days",0),last_commit_days=f.get("last_commit_days",0),total_contributors=f.get("total_contributors",0),topics=", ".join(f.get("topics",[])) if f.get("topics") else "None",primary_language=f.get("primary_language","Unknown"),has_ci="Yes" if f.get("has_ci") else "No",has_tests="Yes" if f.get("has_tests") else "No",has_license="Yes" if f.get("has_license") else "No",readme_size=f.get("readme_size",0),readme_for_prompt=rp)
    try:
        c=genai.Client(api_key=k)
        resp=c.models.generate_content(model="gemini-3.6-flash",contents=pt,config=types.GenerateContentConfig(temperature=0.3,max_output_tokens=500))
        if resp.text:return{"review":resp.text.strip(),"status":"success","provider":"gemini"}
        else:return{"review":"AI review unavailable: Empty response.","status":"error","provider":"gemini"}
    except Exception as e:return{"review":"AI review unavailable: "+str(e),"status":"error","provider":"gemini"}
def generate_ai_review(readme_content:str,features:dict,prediction:int,probability:float)->dict:
    if not readme_content or not readme_content.strip():return{"review":"AI review unavailable: README empty.","status":"error","provider":"none"}
    nres=N(readme_content,features,prediction,probability)
    if nres["status"]=="success":return nres
    gres=g_(readme_content,features,prediction,probability)
    if gres["status"]=="success":return gres
    else:gres["review"]="AI review unavailable: Both NVIDIA and Gemini failed.\nNVIDIA: "+nres["review"]+"\nGemini: "+gres["review"];return gres
def format_ai_review_for_display(ai_review_result:dict)->str:
    if ai_review_result.get("status")=="success":prov=ai_review_result.get("provider","unknown").upper();return"**AI Review** (via {}):\n\n{}".format(prov,ai_review_result.get('review',''))
    elif ai_review_result.get("status")=="skipped":return"*AI review skipped: {}*".format(ai_review_result.get('review',''))
    else:return"*AI review unavailable: {}*".format(ai_review_result.get('review',''))