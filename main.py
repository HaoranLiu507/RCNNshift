import RCNNshift as tracker
import os

# Video path and name
path = os.path.abspath('videos')
name = 'blurbody'

# Tracker options: RCNNshift, RCNNshift_3D or meanshift
select_tracker = 'RCNNshift'

# Select the ROI window in the first frame, options: mouse, input, or batch_input
select_rect = 'mouse'

# Perform options: live or local
# If 'live', present tracking in real-time;
# If 'local', save the tracking results as a video file, then display it
perform = 'live'

# Initialize tracker
Tracker = tracker.RCNNshift(weight=38,batch_size=50, select_tracker=select_tracker, perform=perform, isColor=True)


# Start tracking
Tracker.track(video_path=path, name=name, select_rect=select_rect, ROI_region=None)