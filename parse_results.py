# Collects every saved run into summary.csv and draws chart 2 (latency vs throughput).
# Run on your laptop after copying results/ off the GPU machine.
import glob, json, os
import pandas as pd
import matplotlib.pyplot as plt

rows = []
for f in sorted(glob.glob("results/*/c*.json")):
    d = json.load(open(f))
    rows.append({
        "run": os.path.basename(os.path.dirname(f)),
        "concurrency": int(os.path.basename(f)[1:-5]),
        "out_tok_s": round(d.get("output_throughput", 0), 1),
        "total_tok_s": round(d.get("total_token_throughput", 0), 1),
        "ttft_ms": round(d.get("median_ttft_ms", 0), 1),
        "tpot_ms": round(d.get("median_tpot_ms", 0), 2),
        "p99_tpot_ms": round(d.get("p99_tpot_ms", 0), 2),
    })

df = pd.DataFrame(rows).sort_values(["run", "concurrency"])
df.to_csv("summary.csv", index=False)
print(df.to_string(index=False))

# chart 2: latency vs throughput for the three main Qwen3-14B chat runs on the EPYC
main_runs = [f"ep_14b_{layout}_in1024_out256" for layout in ("tp4", "pp4", "tp2pp2")]
sel = df[df.run.isin(main_runs)]
for run, g in sel.groupby("run"):
    plt.plot(g.tpot_ms, g.out_tok_s, marker="o", label=run.split("_in")[0])
plt.xlabel("median time per output token (ms)")
plt.ylabel("output tokens/s")
plt.legend()
plt.grid(alpha=.3)
plt.savefig("latency_vs_throughput.png", dpi=150)
