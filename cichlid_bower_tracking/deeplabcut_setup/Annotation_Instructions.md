walkthrough content on how to set up the GUI locally (with docker and such)
  * https://github.com/Human-Augment-Analytics/CichlidBowerTracking/tree/master/cichlid_bower_tracking/deeplabcut_setup
How to Label the videos
  * consistently label similar spots
  * Label the parts of the fish that are visible
  * label the nose at the front of the fish
  * label the fins at the point they connect to the body
  * label spine-1 between the eyes
  * label spine-2 between the fins
  * label spine 3 about halfway down the taper
  * label spine 4 at the end of the fish
  * Label the reflections as if the reflection was it's own fish and not a reflection (so the reflections labeled left fin should be the real fish's right fin, etc.)
Deeplabcut interface
  * read through the user guide here:
  * https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html
  * right click to label
  * left click to select a label and move/adjust it
  * center click (clicking in mouse wheel) to remove a label
Videos to label
  * need 1000 labeled frames per species
  * Each folder contains trial folders, within each trial folder is a Videos folder that will contain the videos that need to have frames extracted and labeled.
  best way to extract frames is pan through a video and then use this script to crop the video to that time frame, and then have deeplabcut extract the frames from   the shorter videos
  * https://github.com/athomas125/ImageProcessing/blob/master/src/clip_to_time_video.py
  MC Multi
    * https://www.dropbox.com/home/BioSci-McGrath/Apps/CichlidPiData/__ProjectData/MC_multi
  OrangeCap
    * https://www.dropbox.com/home/BioSci-McGrath/Apps/CichlidPiData/__ProjectData/OC_Build
  YellowHead
    * https://www.dropbox.com/home/BioSci-McGrath/Apps/CichlidPiData/__ProjectData/YH_Build

