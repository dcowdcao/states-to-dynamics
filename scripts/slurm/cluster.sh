# Every job and submitter sources this for the cluster settings in cluster.env.
cluster_env="${SPADE_REPO:?}/scripts/slurm/cluster.env"
if [ ! -f "$cluster_env" ]; then
    echo "error: $cluster_env is missing. Copy scripts/slurm/cluster.env.example to it and set your account, partitions and environment." >&2
    exit 1
fi
source "$cluster_env"
