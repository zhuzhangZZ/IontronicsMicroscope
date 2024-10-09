import nptdms
#import matplotlib.pyplot as plt

filepath = "/Users/sanli/Repositories/KukuraLab/data/Aug_ITO/01/ITOs2_gnp15_vh1p4V_vln0p4_per10sec_0/cam1/"

for i in range(4):
    filename = "event" + str(i) +".tdms"
    file = filepath + filename
    tdms_file = nptdms.TdmsFile(file)
    attributes = tdms_file.object("img", 'cam1').properties
    print(i, attributes['Image size'], attributes['Image size 2'])

#data_raw = tdms_file.channel_data('img', 'cam1')
# #group and channel name depends on labview program. use tdms_file.groups() and tdms_file.group_channels()
# print(data_raw.shape)
# Lx = 102   #the frame size should be read from the attrbutes
# N = data_raw.size//(Lx*Lx) #number of recorded frames
# data = data_raw.reshape(N, Lx, Lx)
# print(data.shape)
# plt.imshow(data[1,:,:])
# plt.show()
