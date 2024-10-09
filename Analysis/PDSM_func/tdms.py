import os
import numpy as np

import nptdms

#from iscat.log import log_info, log_debug, log_warning, log_and_raise_error

def read_tdms(filename, verbose=False, do_remove_empty_frames=True, do_check_no_duplicates=True, do_check_saturation=False, max_duplicate_distance=10, output_attributes=False):
    f = nptdms.TdmsFile(filename)
    data_raw = f.channel_data('img', 'cam2')
    attributes = f.object("img" , 'cam2').properties
    if verbose:
        print(attributes)
    Ly = int(attributes['Image size'])
    Lx = Ly if 'Image size 2' not in attributes else int(attributes['Image size 2'])
    N = data_raw.size//(Ly*Lx)
    data = data_raw[:N*Lx*Ly].reshape(N, Ly, Lx)
    if do_remove_empty_frames:
        data = remove_empty_frames(data)
    if do_check_no_duplicates:
        for k in range(1,max_duplicate_distance+1):
            i_duplicates = ((abs(data[k:,:,:]-data[:-k,:,:])).sum(axis=1).sum(axis=1).min() == 0)
            no_duplicates = i_duplicates.sum()
            if no_duplicates > 0:
                print('Duplicate frame(s) in movie detected: distance: %i; frames: %i/%i; positions: %s)' % (k, no_duplicates, len(data), str(np.where(i_duplicates)[0])))
    if do_check_saturation:
        saturation_level = 4000
        saturated_pix = data.max(axis=0) > saturation_level
        n_saturated_pix = saturated_pix.sum()
        if n_saturated_pix > 0:
            print('Saturated pixels detected! %i pixels exceed the saturation level (%i arb. detector units).' % (n_saturated_pix, saturation_level))
    if output_attributes:
        return data, attributes
    else:
        return data

def read_tdms_attributes(filename):
    f = nptdms.TdmsFile(filename)
    attributes = f.object("img", 'cam2').properties
    return attributes

    
def remove_empty_frames(data):
    s = data.sum(axis=1).sum(axis=1) > 0
    if s.all():
        return data
    else:
        return data[s]
