## File usages:
  * `detect_trans.py`: script to automatically detect the transition scenes and clip the original videos accordingly.
  * `random_select.py`: randomly select clipped videos
  * `stats.py`: calculate the stats of the selected videos
  * `process_manual.py`: read the `manual_clip.txt` file and clip the original videos accordingly
  * `manual_clip.txt`: video clips that we select manually.

## Procedures:
  * First clip the original videos into short clips by PySceneDetect (a command-line tool and Python library, which uses OpenCV to analyze a video to find scene changes or cuts. https://github.com/Breakthrough/PySceneDetect)
  * Then we concatenate those short clips together so that it exceeds 1 min or it is towards the end (the last part could be shorter than 1 min if video ends there)
  * Selection process of the clips:
    * One clip for each original video is selected by the script, we will manually check:
        * whether the video contains information we want 
        * \>= 30 seconds and <= 4 minutes 
    * If we do not want the clip, we delete it and run the script again to randomly select another.
    * We iterate the first two steps until there is no more generated clips (We then manually select a duration in the original video for those videos with no clip selected during the process)

