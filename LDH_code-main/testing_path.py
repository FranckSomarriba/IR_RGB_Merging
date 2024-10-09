import os
from PyQt5.QtCore import QLibraryInfo, QLibraryInfo

# # Get the plugin path from PyQt5
qt_plugin_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)

# # Set the environment variable dynamically
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path


# Set the environment variable for Qt plugin path to prevent compatibility issues
# Qt is a cross-platform software for GUI related functions
#os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/franck/libraries/anaconda3/lib/python3.11/site-packages/cv2/qt/plugins'

print (os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'])