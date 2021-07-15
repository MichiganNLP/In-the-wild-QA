# Randomly select videos links from each channel

import random
import os

random.seed(42)

TOTAL_NUM = 100 # total video number in domain
CHANNEL_NUM = 8 # channel number for the domain
DOMAIN = 'Military' # domain name
CHANNEL_NAMES = ["Gung-Ho-Vids", "AiirSource-Military", "Army-military-2018", 
"WarLeaks-Military-Blog", "Military-Archive", "USA-Military-Channel", 
"MilitaryNotes", "Sandboxx"] # channel names under that domain


def select(name, num, iteration=0):
    if iteration == 0:
        with open(f'All-links/{DOMAIN}/{name}.txt', 'r') as f:
            links = f.readlines()
    else:
        with open(f'Selected-links/{DOMAIN}/{name}-{iteration - 1}-others.txt', 'r') as f:
            links = f.readlines()
    
    sel_links = random.sample(links, num)
    other_links = [l for l in links if l not in sel_links]
    
    return sel_links, other_links

ITERATION = 18

# second to later iteration, how many videos need to be substituted
SUBS_NUM = {
    "Gung-Ho-Vids": 0, 
    "AiirSource-Military": 0, 
    "Army-military-2018": 0,
    "WarLeaks-Military-Blog": 0, 
    "Military-Archive": 0, 
    "USA-Military-Channel": 10, 
    "MilitaryNotes":0 , 
    "Sandboxx": 10
}

if __name__ == "__main__":
    # channel index that we will select one more video
    ch1 = random.sample(CHANNEL_NAMES, TOTAL_NUM % CHANNEL_NUM)

    for name in CHANNEL_NAMES:

        if ITERATION == 0:
            num = TOTAL_NUM // CHANNEL_NUM
            if name in ch1:
                num += 1
        else:
            assert name in SUBS_NUM
            num = SUBS_NUM[name]
        
        sel_links, other_links = select(name, num, iteration=ITERATION)

        if not os.path.exists(f'Selected-links/{DOMAIN}'):
            os.makedirs(f'Selected-links/{DOMAIN}')
        
        with open(f'Selected-links/{DOMAIN}/{name}-{ITERATION}-selected.txt', 'w') as f:
            f.write(''.join(sel_links))
        with open(f'Selected-links/{DOMAIN}/{name}-{ITERATION}-others.txt', 'w') as f:
            f.write(''.join(other_links))
