import os
import json


def main():
    with open('dr_video_ids.txt', 'r') as f:
        data = f.readlines()
    data = data[3:]
    drive_ids = dict()  # google drive ids for each video clip
    for d in data:
        dr_id, name, _, _ = d.split()
        drive_ids[name] = dr_id
    
    v_types = dict()    # domain for each video clip
    clip_rt = "../../video_clipping/selected_clips/"
    dms = os.listdir(clip_rt)
    for dm in dms:
        chs = os.listdir(os.path.join(clip_rt, dm))
        for ch in chs:
            v_names = os.listdir(os.path.join(clip_rt, dm, ch))
            for v_name in v_names:
                v_types[v_name] = dm

    # examine whether google drive video clips are the same as what we have locally
    # NOTE: we manually modified the results returned by find_url_for_drive.py
    for k, _ in v_types.items():
        assert k in drive_ids, f"{k} in original clips but not in drive"
    for k, _ in drive_ids.items():
        assert k in v_types, f"{k} on drive but not in original clips"
    
    # get the original video link as well as the descriptions for those clips
    ov_infos = dict()
    ov_rt = "../../youtube-crawler/Description/"
    for dm in dms:
        chs = os.listdir(os.path.join(ov_rt, dm))
        for ch in chs:
            with open(os.path.join(ov_rt, dm, ch), 'r') as f:
                raw_data = f.readlines()
            data = [json.loads(d) for d in raw_data]
            for i, itm in enumerate(data):
                ov_infos[ch.split('.json')[0] + "_" + str(i)] = itm
    
    v_se = dict()
    # get the selected video clips start and end time

    manual_info = dict()
    with open("../../video_clipping/manual_clip.txt", 'r') as f:
        manual_data = f.readlines()
    for d in manual_data:
        toks = d.split()
        if len(toks) <= 1:
            continue
        ov_name, sc_start, sc_end = toks[0], toks[1], toks[3]
        manual_info[ov_name] = {
            "seconds": {
                "start": sc_start,
                "end": sc_end
            }
        }

    for name, _ in v_types.items():
        if "-clip-" in name:
            ov_n, clip_i = name.split("-clip-")
            clip_i = int(clip_i.split(".mp4")[0])
            ch = ov_n.split("_")[0]
            with open(f"../../video_clipping/auto-clips-info/{v_types[name]}/{ch}/{ov_n}-frames.txt", 'r') as f:
                raw_data = f.readlines()
            frm_start, frm_end = raw_data[clip_i].split("-")
            with open(f"../../video_clipping/auto-clips-info/{v_types[name]}/{ch}/{ov_n}-seconds.txt", 'r') as f:
                raw_data = f.readlines()
            sc_start, sc_end = raw_data[clip_i].split("-")
            v_se[name] = {
                "frames" : {
                    "start": frm_start,
                    "end": frm_end
                },
                "seconds": {
                    "start": sc_start,
                    "end": sc_end
                },
                "split-method": "auto"
            }
        else:
            assert "-manual" in name
            ov_n = name.split("-manual")[0]
            assert ov_n in manual_info
            v_se[name] = {
                "seconds": {
                    "start": manual_info[ov_n]["seconds"]["start"],
                    "end": manual_info[ov_n]["seconds"]["end"]
                },
                "split-method": "manual"
            }
    
    # write all the metadata to an output file
    info = list()
    for name, dr_id in drive_ids.items():
        channel = name.split("_")[0]
        if "-clip-" in name:
            ov_name = name.split("-clip-")[0]
        else:
            assert "-manual" in name
            ov_name = name.split("-manual")[0]

        info.append({
            "video_name": name,
            "google_drive_id": dr_id,
            "domain": v_types[name],
            "channel": channel,
            "url_for_original_video": ov_infos[ov_name]["url"],
            "description": ov_infos[ov_name]["description"],
            "time_in_original_video": v_se[name]
        })

    with open("general_info.json", 'w') as f:
        json.dump(info, f, sort_keys = False, indent = 4,
               ensure_ascii = False)

if __name__ == "__main__":
    main()