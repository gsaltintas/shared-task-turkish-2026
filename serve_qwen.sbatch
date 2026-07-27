#!/bin/bash
# Qwen3.6-35B-A3B'yi nsdl2/H200'de tek GPU ile vLLM sunucusu olarak kaldırır,
# hazır olunca smoke-test yapar, openai_csv_runner ile data.tsv'yi koşar ve
# analyze_outputs ile skorlar. (Repo koduna dokunmaz; sadece bizim runner + hazır scorer.)
#
# Kullanım:   cd <repo> && sbatch serve_qwen.sbatch
# Test için:  NUM=5 sbatch serve_qwen.sbatch          (ilk 5 soru)
# Server'ı açık tut: KEEP_ALIVE=1 sbatch serve_qwen.sbatch
#
#SBATCH --job-name=qwen36-serve
#SBATCH --partition=slurm_queue
#SBATCH --nodelist=nsdl2
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/user/home/t.tuna/sharedtask/shared-task-turkish-2026/logs/qwen36_%j.out
set -uo pipefail

REPO=/user/home/t.tuna/sharedtask/shared-task-turkish-2026
cd "$REPO" || exit 1
mkdir -p logs output

# --- ayarlar (env ile override edilebilir) ---
SERVER_ENV="${SERVER_ENV:-evalhub_env}"          # fallback: vllm_gemma4
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"           # yüklenecek HF modeli
SERVED_NAME="${SERVED_NAME:-qwen3.6}"            # API model adı + kolon etiketi (-> "qwen3.6 Response")
PORT="${PORT:-8765}"
MAXLEN="${MAXLEN:-32768}"
EVAL_PYTHON="${EVAL_PYTHON:-/opt/Anaconda-2021.05/bin/python3}"   # openai_csv_runner: requests var, pandas gerekmez
OUT="${OUT:-output/output_qwen36.tsv}"
NUM="${NUM:-}"                                    # NUM=5 -> sadece ilk 5 satır
LOG="logs/vllm_${SLURM_JOB_ID:-local}.log"

# --- server ortamı ---
source /opt/Anaconda-2021.05/etc/profile.d/conda.sh
conda activate "$SERVER_ENV" || { echo "conda activate $SERVER_ENV başarısız" >&2; exit 1; }

echo "[$(date +%T)] vLLM başlatılıyor: $MODEL  (served=$SERVED_NAME)  :$PORT  env=$SERVER_ENV  maxlen=$MAXLEN"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization 0.92 --dtype bfloat16 \
  --trust-remote-code --max-model-len "$MAXLEN" \
  > "$LOG" 2>&1 &
SERVER_PID=$!

# --- hazır olmayı bekle (max ~20 dk) ---
echo "[$(date +%T)] /v1/models bekleniyor..."
READY=0
for i in $(seq 1 120); do
  if curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
    READY=1; echo "[$(date +%T)] server HAZIR (deneme $i)"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "HATA: server süreci öldü. Son log:" >&2; tail -n 60 "$LOG" >&2; exit 1
  fi
  sleep 10
done
[ "$READY" = "1" ] || { echo "HATA: server zamanında hazır olmadı. Son log:" >&2; tail -n 60 "$LOG" >&2; kill "$SERVER_PID" 2>/dev/null; exit 1; }

# --- smoke test ---
echo "[$(date +%T)] smoke test:"
curl -s "http://localhost:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Tek cümlede: Türkiye'nin başkenti neresidir?\"}]}"
echo

# --- generation (bizim runner) ---
echo "[$(date +%T)] openai_csv_runner çalışıyor -> $OUT"
EXTRA=()
[ -n "$NUM" ] && EXTRA=(--num-examples "$NUM")
"$EVAL_PYTHON" scripts/openai_csv_runner.py data/data.tsv \
  --base-url "http://localhost:$PORT/v1" --api-key-env NONE \
  --model "$SERVED_NAME" -o "$OUT" "${EXTRA[@]}"
RC=$?

# --- skorlama (hazır analyze_outputs, AYNEN) ---
if [ $RC -eq 0 ]; then
  echo "[$(date +%T)] analyze_outputs skoru:"
  "$EVAL_PYTHON" scripts/analyze_outputs.py "$OUT"
fi
echo "[$(date +%T)] bitti. runner rc=$RC  ->  $OUT"

# --- kapanış ---
if [ "${KEEP_ALIVE:-0}" = "1" ]; then
  echo "[$(date +%T)] KEEP_ALIVE=1 — server açık kalıyor (scancel ile durdur)."
  wait "$SERVER_PID"
else
  kill "$SERVER_PID" 2>/dev/null || true
fi
exit $RC
