# Build the job environments on node-local disk and pack them into .cache/envs (scripts/build_env_tar.sh).
set -euo pipefail
timeout 20 curl -sI https://pypi.org/simple/ | head -1 || { echo "!!! no network on this node"; exit 1; }
export BUILD_ROOT=$PBS_LOCALDIR/envbuild
for kind in ${KINDS:-main ref}; do
  echo "=== $kind"
  $REPO/scripts/build_env_tar.sh $kind
done
