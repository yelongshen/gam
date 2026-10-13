import re
lines = open("/tmp/still_by_run.log").read().splitlines()
for ln in lines:
    ln = re.sub(r"\s+", " ", ln).strip()
    print(ln[:230])
