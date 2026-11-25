"""Small interactive browser for generated explanations and judge outputs."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"
st.set_page_config(page_title="XAI – décrochage",layout="wide")
st.title("Comprendre une prédiction — prototype pédagogique")
st.warning("Une prédiction n'est ni un verdict ni une causalité. Ne prenez aucune décision à partir de cet outil seul.")
def rows(name):
 p=OUT/name
 return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []
inst_path=OUT/"selected_instances.csv"
if not inst_path.exists(): st.info("Exécutez d'abord `python -m src.evaluate`."); st.stop()
instances=pd.read_csv(inst_path); explanations=rows("xai_explanations.jsonl"); answers=rows("llm_responses.jsonl"); scores=rows("judge_scores.jsonl")
ids=instances["instance_id"].tolist(); iid=st.sidebar.selectbox("Instance test",ids)
methods={"Attributions SHAP":"shap","Règle Anchors":"anchors","Contrefactuel":"counterfactual"}
method_label=st.sidebar.selectbox("Explication technique",list(methods)); method=methods[method_label]
providers=sorted({x["provider"] for x in answers}); provider=st.sidebar.selectbox("LLM générateur",providers or ["(aucune réponse)"])
levels=["N1","N2","N3"]; level=st.sidebar.selectbox("Niveau",levels)
profile=None
if level=="N3": profile=st.sidebar.selectbox("Profil",["student","family","teacher"])
row=instances.loc[instances.instance_id==iid].iloc[0]; xai=next((x for x in explanations if x["instance_id"]==iid),None)
a,b=st.columns(2)
with a:
 st.subheader("Prédiction")
 st.write({"Classe réelle":row.true_class,"Classe prédite":row.predicted_class,"Confiance max":round(float(row.confidence),3),"Correcte":bool(row.is_correct)})
 st.json({c:float(row[f"probability_{c}"]) for c in ["Dropout","Enrolled","Graduate"]})
with b:
 st.subheader("Sortie XAI")
 if xai: st.json(xai[method])
 else: st.info("Aucune sortie XAI. Exécutez python -m src.explain.")
answer=next((x for x in answers if x["instance_id"]==iid and x["method"]==method and x["provider"]==provider and x["level"]==level and x["profile"]==profile),None)
st.subheader("Explication en langage naturel")
st.write(answer["response"] if answer else "Aucune réponse enregistrée pour ces choix.")
if answer:
 score=next((s for s in scores if s["cache_key"]==answer["cache_key"]),None)
 st.subheader("Évaluation du juge (à contrôler manuellement)")
 if score: st.json(score["evaluation"])
 else: st.caption("Aucune évaluation du juge disponible.")
st.divider(); st.subheader("Synthèse des évaluations disponibles")
if scores:
 aggregate=[]
 for s in scores:
  ev=s["evaluation"]
  if "scores" not in ev: continue
  for criterion,item in ev["scores"].items(): aggregate.append({"Critère":criterion,"Note":item["score"]})
 if aggregate: st.bar_chart(pd.DataFrame(aggregate).groupby("Critère")["Note"].mean())
 else: st.info("Aucun score valide.")
else: st.info("Les sorties du juge s'afficheront après `python -m src.llm_pipeline judge`.")
