#!/usr/bin/env bash
# Run on the LAPTOP. Starts / connects to / stops a Jupyter server on a MATS-cluster GPU.
#
#   cluster/jupyter.sh start                 # 1 L40, debug QoS, 2 h hard limit (default)
#   cluster/jupyter.sh start normal 06:00:00 # normal QoS, given walltime (<= 24 h); holds a shared GPU
#   cluster/jupyter.sh connect               # wait for the server, open the tunnel, print the URL (Ctrl-C closes)
#   cluster/jupyter.sh status                # your jobs + the current session file (token hidden)
#   cluster/jupyter.sh stop                  # scancel the session's job
#
# Env: COMMUTE_CLUSTER (default e.agnew@185.141.218.207), COMMUTE_CLUSTER_REPO (default the lead clone), LOCAL_PORT (default: same as the remote port).
set -euo pipefail

HOST=${COMMUTE_CLUSTER:-e.agnew@185.141.218.207}
REPO=${COMMUTE_CLUSTER_REPO:-/mnt/nw/home/e.agnew/commute-agents/lead}   # a clone on the cluster
SESSION=$REPO/.jupyter-session

session() { ssh -o BatchMode=yes "$HOST" "cat $SESSION 2>/dev/null" || true; }
field() { sed -n "s/^$1=//p" <<<"$2"; }

case "${1:-}" in
start)
    qos=${2:-debug}
    args=(--qos="$qos")
    if [[ $qos == normal ]]; then
        args+=(--time="${3:?give a walltime for normal QoS, e.g. 06:00:00}")
    elif [[ -n "${3:-}" ]]; then
        args+=(--time="$3")
    fi
    id=$(ssh -o BatchMode=yes "$HOST" "cd $REPO && sbatch --parsable ${args[*]} cluster/jupyter.sbatch")
    echo "submitted job $id (${args[*]}). Next: cluster/jupyter.sh connect   Stop: cluster/jupyter.sh stop"
    ;;
connect)
    echo "waiting for the server (Ctrl-C to give up; the job keeps running)..."
    while :; do
        s=$(session)
        id=$(field JOB_ID "$s")
        if [[ -n $id ]]; then
            state=$(ssh -o BatchMode=yes "$HOST" "squeue -h -j $id -o %T 2>/dev/null" || true)
            [[ $state == RUNNING ]] && break
        fi
        ssh -o BatchMode=yes "$HOST" "squeue -h -u \$USER -n commute-jupyter -o '  job %i %T %r'" || true
        sleep 10
    done
    node=$(field NODE "$s"); port=$(field PORT "$s"); token=$(field TOKEN "$s")
    end=$(ssh -o BatchMode=yes "$HOST" "squeue -h -j $id -o %e")
    lport=${LOCAL_PORT:-$port}
    cat <<EOF

Job $id on $node, server port $port, ends $end (cluster time).
Jupyter URL (VS Code: Select Kernel -> Existing Jupyter Server -> paste):

    http://localhost:$lport/?token=$token

Tunnel open; keep this terminal running. Ctrl-C closes the tunnel (the job keeps running:
cluster/jupyter.sh stop cancels it).
EOF
    # The node's ports are firewalled from the controller, so enter the node through a CPU-only step of the
    # job itself (srun relays stdio to the node's sshd on localhost), then forward to the server's localhost port.
    relay="srun --jobid=$id --overlap --gres=none --unbuffered --ntasks=1 --cpus-per-task=1 --mem=100M nc 127.0.0.1 22"
    exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
        -o HostKeyAlias=mats-l40-worker -o StrictHostKeyChecking=accept-new \
        -o ProxyCommand="ssh -o BatchMode=yes $HOST $relay" \
        -L "$lport:127.0.0.1:$port" "${HOST%@*}@$node"
    ;;
status)
    ssh -o BatchMode=yes "$HOST" "squeue -u \$USER -o '%.8i %.16j %.7q %.9T %.10M %.10l %.10L %R'"
    session | grep -v '^TOKEN=' || echo "(no session file)"
    ;;
stop)
    id=${2:-$(field JOB_ID "$(session)")}
    [[ -n $id ]] || { echo "no session file; give the job id: cluster/jupyter.sh stop <id>"; exit 1; }
    ssh -o BatchMode=yes "$HOST" "scancel $id && squeue -h -j $id -o '%i %T' 2>/dev/null; true"
    echo "cancelled job $id"
    ;;
*)
    sed -n '2,10p' "$0"; exit 1 ;;
esac
