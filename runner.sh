#bash runnerfile

## if run in parallel
np=5
cat run.sh | xargs -L 1 -I CMD -P ${np} bash -c CMD
