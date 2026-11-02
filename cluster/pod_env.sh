# Source me on a pod: the machine's own variables, for this shell and its children.
# On Vast the plugin's settings (inherit-env, [env]) and Vast's CONTAINER_* reach PID 1 only, not the Jupyter kernel, and
# the kernel's UV_CACHE_DIR is the image's (seen 2026-11-02). On RunPod this changes nothing that matters. Names listed
# here only; never print a value.
while IFS= read -r -d '' e; do
  case "${e%%=*}" in
    WANDB_API_KEY|GIT_PAT|COMMUTE_CACHE_DIR|HF_HOME|UV_CACHE_DIR|RUNPOD_POD_ID|RUNPOD_API_KEY|CONTAINER_ID|CONTAINER_API_KEY) export "$e" ;;
  esac
done < /proc/1/environ
