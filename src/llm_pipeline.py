"""Reproducible, cached LLM generation and blind-judge pipeline."""
from __future__ import annotations
import hashlib, json, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; PROMPTS=ROOT/"prompts"
PROVIDERS={"groq":("GROQ_API_KEY","GROQ_BASE_URL","GROQ_MODEL"),"mistral":("MISTRAL_API_KEY","MISTRAL_BASE_URL","MISTRAL_MODEL"),"gemini":("GEMINI_API_KEY","GEMINI_BASE_URL","GEMINI_MODEL")}
PROFILES={"student":"ÉTUDIANT : langage simple, bienveillant et direct; proposer des pistes réalistes à discuter avec l'équipe pédagogique. Ne pas culpabiliser ni prédire un destin.","family":"FAMILLE : langage clair, rassurant et sans jargon; expliquer comment écouter et accompagner sans pression ni accusation.","teacher":"ENSEIGNANT RÉFÉRENT : synthèse professionnelle et leviers d'accompagnement concrets à vérifier avec l'étudiant; ne pas transformer le score en décision."}
DOMAIN="Âge, genre, nationalité, parcours antérieur et diplômes parentaux sont immuables. Les notes d'admission sont bornées (0–200), les notes de semestre (0–20); les unités validées ne peuvent dépasser les unités évaluées/inscrites. Les codes catégoriels sont des identifiants, pas des quantités : ne pas interpréter leur ordre. Les frais à jour (0=non, 1=oui) peuvent être discutés avec le service social; les résultats scolaires sont des observations, pas des causes. N'inventez aucun lien causal. Le modèle est un outil d'aide, non un verdict."

def read(name): return (PROMPTS/name).read_text(encoding="utf-8")
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def load_jsonl(p):
    if not p.exists(): return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
def append(p,obj):
    with p.open("a",encoding="utf-8") as f: f.write(json.dumps(obj,ensure_ascii=False)+"\n")
def invoke(provider, model, base, key, messages, temperature, version):
    client=OpenAI(api_key=key,base_url=base)
    for attempt in range(5):
        try:
            response=client.chat.completions.create(model=model,messages=messages,temperature=temperature)
            return response.choices[0].message.content or ""
        except Exception:
            if attempt==4: raise
            time.sleep(min(60,2**attempt+random.random()) )
def main():
    load_dotenv(ROOT/".env"); OUT.mkdir(exist_ok=True)
    mode=sys.argv[1] if len(sys.argv)>1 else "generate"
    if mode=="generate":
        explanations=load_jsonl(OUT/"xai_explanations.jsonl")
        if not explanations: raise SystemExit("Exécutez d'abord python -m src.explain")
        responses=load_jsonl(OUT/"llm_responses.jsonl"); cache={r["cache_key"]:r for r in responses}
        for e in explanations:
            for method in ("shap","anchors","counterfactual"):
                for provider,(key_env,base_env,model_env) in PROVIDERS.items():
                    key=os.getenv(key_env); base=os.getenv(base_env); model=os.getenv(model_env)
                    if not all((key,base,model)): print(f"Ignore {provider}: config incomplète"); continue
                    levels=[("N1",None)]+[("N2",None)]+[("N3",p) for p in PROFILES]
                    for level,profile in levels:
                        version="v1"; system=read("generator_system.txt"); user_template=read("generator_user.txt")
                        payload={"instance_id":e["instance_id"],"prediction":e["predicted_class"],"probabilities":e["probabilities"],"explanation":e[method],"method":method}
                        user=user_template.format(level=level,domain=(DOMAIN if level in ("N2","N3") else "Non fourni."),profile=(PROFILES[profile] if profile else "Aucun profil spécifique."),payload=json.dumps(payload,ensure_ascii=False))
                        messages=[{"role":"system","content":system},{"role":"user","content":user}]
                        ck=digest({"provider":provider,"model":model,"temperature":0.2,"version":version,"messages":messages})
                        if ck in cache: continue
                        answer=invoke(provider,model,base,key,messages,0.2,version)
                        record={"cache_key":ck,"instance_id":e["instance_id"],"method":method,"level":level,"profile":profile,"provider":provider,"model":model,"temperature":0.2,"prompt_version":version,"timestamp":datetime.now(timezone.utc).isoformat(),"messages":messages,"response":answer}
                        append(OUT/"llm_responses.jsonl",record); append(OUT/"llm_calls.jsonl",record); cache[ck]=record
                        print(f"{provider} {e['instance_id']} {method} {level} {profile or ''}")
    elif mode=="judge":
        rows=load_jsonl(OUT/"llm_responses.jsonl"); existing=load_jsonl(OUT/"judge_scores.jsonl"); done={r["cache_key"] for r in existing}
        key=os.getenv("JUDGE_API_KEY"); base=os.getenv("JUDGE_BASE_URL"); model=os.getenv("JUDGE_MODEL")
        if not all((key,base,model)): raise SystemExit("Configurez JUDGE_API_KEY, JUDGE_BASE_URL et JUDGE_MODEL")
        refs={r["instance_id"]:r for r in load_jsonl(OUT/"xai_explanations.jsonl")}
        system=read("judge_system.txt")
        for r in rows:
            if r["cache_key"] in done: continue
            ref=refs[r["instance_id"]]
            # Le niveau et le fournisseur/modèle ne sont délibérément pas transmis au juge.
            user=read("judge_user.txt").format(reference=json.dumps({"instance":r["instance_id"],"prediction":ref["predicted_class"],"xai":ref[r["method"]],"domain":DOMAIN,"profile":PROFILES.get(r["profile"],"Destinataire général")},ensure_ascii=False), explanation=r["response"])
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
            answer=invoke("judge",model,base,key,messages,0.0,"judge-v1")
            try:
                parsed=json.loads(answer); assert all(1<=int(parsed["scores"][k]["score"])<=5 for k in ("faithfulness","hallucination","domain","profile_fit","actionability")); assert all(c["verdict"] in ("correct","incorrect","unverifiable") for c in parsed["claims"])
                verifiable=[c for c in parsed["claims"] if c["verdict"]!="unverifiable"]
                parsed["coherence"]=(sum(c["verdict"]=="correct" for c in verifiable)/len(verifiable)) if verifiable else None
            except Exception as exc: parsed={"validation_error":str(exc),"raw_response":answer}
            result={"cache_key":r["cache_key"],"instance_id":r["instance_id"],"method":r["method"],"blind":True,"judge_model":model,"timestamp":datetime.now(timezone.utc).isoformat(),"evaluation":parsed}
            append(OUT/"judge_scores.jsonl",result); append(OUT/"llm_calls.jsonl",{"provider":"judge","model":model,"temperature":0.0,"prompt_version":"judge-v1","messages":messages,"response":answer,"timestamp":result["timestamp"],"cache_key":r["cache_key"]})
    else: raise SystemExit("Usage: python -m src.llm_pipeline [generate|judge]")
if __name__=="__main__": main()
