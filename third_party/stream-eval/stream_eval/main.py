import argparse
import re
import json
from tqdm.contrib.concurrent import thread_map
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from model_hub import ModelHub
from json_repair import repair_json

from stream_eval.prompt.judger_prompt import judger_prompt
from stream_eval.prompt.placeholder_prompt import placeholder_prompt

def parse_args():
    parser = argparse.ArgumentParser(description="StreamEval evaluation")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--judger", type=str, default="Qwen3-235B-A22B", help="judger name")
    parser.add_argument("--model-output", type=str, required=True, help="Path to model output file/dir")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for evaluation output")
    parser.add_argument("--max-workers", type=int, default=8, help="Max concurrent judge calls (default 8; avoid 64 to prevent judge API blocking)")
    return parser.parse_args()

def parse_json(res):
    try:
        return json.loads(repair_json(str(res)))
    except:
        return {}

def load_json_or_jsonl(filename):
    if filename.endswith(".json"):
        with open(filename, "r") as f:
            data = json.load(f)
        return [data]
    else:
        data = []
        with open(filename, "r") as f:
            for line in f.readlines():
                if line:
                    data.append(parse_json(line))
        return data

class StreamEval:
    def __init__(self, model_config: str, judger: str, model_output: str, output_dir: str, max_workers: int = 8):
        self.judger = judger
        self.model_output = model_output
        self.output_dir = output_dir
        self.max_workers = max_workers
        self._hub = ModelHub(model_config)

    def extract_qa_data(self, data):
        qa_entries = {}

        for key, value in data.items():
            match = re.match(r'qa_(\d+)_(.+)', key)
            if not match:
                continue
            idx, field = match.groups()
            idx = int(idx)
            qa_entries.setdefault(idx, {})[field] = value

        results = []
        for idx in sorted(qa_entries):
            entry = qa_entries[idx]

            # Handle typo: 'resposne' vs 'response'. Do not use `or` here:
            # response_timesec can be 0, which is a valid timestamp.
            human_resp = entry.get('response') if entry.get('response') is not None else entry.get('resposne')
            human_time = entry.get('response_timesec') if entry.get('response_timesec') is not None else entry.get('resposne_timesec')

            # Collect model responses
            model_responses = sorted(
                [
                    {"time_sec": int(k.split('_')[-1]), "response": v}
                    for k, v in entry.items()
                    if k.startswith('model_timesec_')
                ],
                key=lambda x: x["time_sec"]
            )

            results.append({
                "question_id": idx,
                "type": entry.get("type"),
                "question_time_sec": entry.get("question_timesec"),
                "question": entry.get("question"),
                "human_response_time_sec": human_time,
                "human_response": human_resp,
                "model_responses": model_responses
            })

        return results

    def is_placeholder(self, answer, question_id=None):
        if str(answer).strip().lower() in [
            "", "silent", "<|silent|>", "<silent>"
        ]:
            return True

        raw = self._hub.call(
            model_name=self.judger,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": placeholder_prompt + answer
                    }
                ]
            }]
        )
        res = raw.get("content", "{}")
        print(f"[Judge placeholder] question_id={question_id}\n  output: {res}")
        res = parse_json(res)
        return res.get("label", "no").lower() == "yes"

    def llm_score(self, question, answer, prediction, question_id=None):
        raw = self._hub.call(
            model_name=self.judger,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": judger_prompt.format(
                            question=question,
                            answer=answer,
                            prediction=prediction
                        )
                    }
                ]
            }]
        )
        res = raw.get("content", "{}")
        print(f"[Judge score] question_id={question_id}\n  output: {res}")
        res = parse_json(res)
        return {
            "score": max(min(res.get("score", 0) * 20, 100), 0),
            "explanation": res.get("explanation", ""),
            "label": "accuracy"
        }

    def run_qa(self, qa):
        question = qa["question"]
        answer = qa["human_response"]
        ht = qa["human_response_time_sec"]
        accept_window = 2 if qa["type"].lower() in ["future", "proactive"] else 0.1
        qid = qa.get("question_id"), qa.get("sample_id")

        for model_res in qa["model_responses"]:
            t = model_res["time_sec"]
            res = model_res["response"]

            model_res["skip"] = self.is_placeholder(res, question_id=qid)
            if model_res["skip"]:
                continue

            if 0 <= (t - ht) <= accept_window:
                return self.llm_score(question, answer, res, question_id=qid)
            elif t < ht:
                return {"score": 0, "explanation": "", "label": "early response"}
            elif t > ht:
                return {"score": 0, "explanation": "", "label": "later response"}

        return {"score": 0, "explanation": "", "label": "no response"}

    def get_sample_id(self, idx, sample):
        return sample.get("rounds", [{}])[0].get("round_id", str(idx))

    def run(self):
        data = load_json_or_jsonl(self.model_output)
        qa_list = [
            {"sample_id": self.get_sample_id(idx, sample)} | qa
            for idx, sample in enumerate(data)
            for qa in self.extract_qa_data(sample.get("vars", {}))
        ]

        results = thread_map(self.run_qa, qa_list, max_workers=self.max_workers)

        qa_list = [qa | res for qa, res in zip(qa_list, results)]
        qa_results = {"results": qa_list}

        # 统计：总样本数、总平均分；按 type 的平均分与比例；按 label 的比例与平均分
        n = len(qa_list)
        scores = [r.get("score", 0) for r in qa_list]
        total_avg = sum(scores) / n if n else 0

        by_type = {}
        for r in qa_list:
            t = r.get("type") or "unknown"
            by_type.setdefault(t, {"count": 0, "scores": []})
            by_type[t]["count"] += 1
            by_type[t]["scores"].append(r.get("score", 0))
        type_stats = {
            t: {
                "count": d["count"],
                "ratio": d["count"] / n if n else 0,
                "avg_score": sum(d["scores"]) / len(d["scores"]) if d["scores"] else 0,
            }
            for t, d in by_type.items()
        }

        by_label = {}
        n_future = 0
        for r in qa_list:
            if r.get("type", "").lower() not in ["future", "proactive"]:
                continue
            n_future += 1
            L = r.get("label") or "unknown"
            by_label.setdefault(L, {"count": 0, "scores": []})
            by_label[L]["count"] += 1
            by_label[L]["scores"].append(r.get("score", 0))
        label_stats = {
            L: {
                "count": d["count"],
                "count_future": n_future,
                "ratio": d["count"] / n_future if n_future else 0,
                "avg_score": sum(d["scores"]) / len(d["scores"]) if d["scores"] else 0,
            }
            for L, d in by_label.items()
        }

        qa_results["statistics"] = {
            "total_count": n,
            "total_avg_score": round(total_avg, 2),
            "by_type": {t: {k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()} for t, s in type_stats.items()},
            "by_label": {L: {k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()} for L, s in label_stats.items()},
        }

        out_file = Path(self.output_dir) / "eval.json"
        with open(out_file, "w") as f:
            json.dump(qa_results, f, indent=2, ensure_ascii=False)

        print(f"Evaluation done: {out_file}")


def main():
    args = parse_args()
    StreamEval(
        model_config=args.model_config,
        judger=args.judger,
        model_output=args.model_output,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    ).run()


if __name__ == "__main__":
    main()
