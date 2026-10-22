import requests, re, json, time
hits=[]; n=0
for page in range(60):
    r=requests.get("https://gateway.polymarket.us/v1/events",params={"active":"true","closed":"false","limit":100,"offset":page*100},timeout=60)
    if r.status_code!=200: print("status",r.status_code,"at page",page); break
    evs=r.json().get("events",[])
    if not evs: break
    for e in evs:
        n+=1
        blob=(e.get("slug","")+" "+str(e.get("title",""))).lower()
        if re.search(r"btc|bitcoin|crypto|ethereum|updown|up-or-down|up or down",blob):
            hits.append((e.get("slug"),e.get("title")))
    time.sleep(0.2)
print("events scanned",n,"crypto-ish hits",len(hits))
for h in hits[:40]: print(h)
