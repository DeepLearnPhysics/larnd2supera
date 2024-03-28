import sys, os
import h5py
import numpy as np
import time
import larnd2supera
import argparse
import ROOT
from edep2supera.utils import get_iomanager, larcv_meta, larcv_particle
from LarpixParser import event_parser as EventParser
from larcv import larcv
import pandas as pd


def get_larnd2supera(config_key):

    driver = larnd2supera.driver.SuperaDriver()
    if os.path.isfile(config_key):
        driver.ConfigureFromFile(config_key)
    else:
        driver.ConfigureFromFile(larnd2supera.config.get_config(config_key))
    
    return driver 

def log_supera_integrity_check(data,driver,log,verbose=False):

    if not log:
        return

    label = driver.Label()
    meta  = driver.Meta()

    # Packet tensor
    pcloud = np.array([[edep.x,edep.y,edep.z,edep.e] for edep in driver._edeps_all])
    voxels = meta.edep2voxelset(driver._edeps_all)
    voxels = np.array([[meta.pos_x(vox.id()),
                        meta.pos_y(vox.id()),
                        meta.pos_z(vox.id()),
                        vox.value()] for vox in voxels.as_vector()
                      ]
                     ) 
    
    cluster_sum = np.sum([p.energy.sum() for p in label.Particles()])
    input_sum  = np.sum([np.sum([edep.e for edep in p.pcloud]) for p in data])
    input_unass= np.sum([edep.e for edep in data.unassociated_edeps])
    energy_sum = label._energies.sum()
    energy_num = label._energies.size()
    pcloud_sum = np.sum(pcloud[:,3])
    pcloud_num = pcloud.shape[0]
    voxels_sum = np.sum(voxels[:,3])
    voxels_num = voxels.shape[0]
    unass_sum  = np.sum([vox.value() for vox in label._unassociated_voxels.as_vector()])
    
    if verbose:
        print('  Raw image    :',voxels_num,'voxels with the sum',voxels_sum)
        print('  Packets      :',pcloud_num,'packets with the sum',pcloud_sum)
        print('  Input cluster:',input_sum)
        print('  Input unass. :',input_unass)
        print('  Label image  :',energy_num,'voxels with the sum',energy_sum)
        print('  Label cluster:',cluster_sum)
        print('  Unassociated :',unass_sum)
        print('  Label image - (Cluster sum + Unassociated)',energy_sum - (cluster_sum + unass_sum))
        print('  Label image - Raw image',energy_sum - voxels_sum)
        print('  Packets - Raw image',pcloud_sum - voxels_sum)

    log['raw_image_sum'].append(voxels_sum)
    log['raw_image_npx'].append(voxels_num)
    log['raw_packet_sum'].append(pcloud_sum)
    log['raw_packet_num'].append(pcloud_num)
    log['in_cluster_sum'].append(input_sum)
    log['in_unass_sum'].append(input_unass)
    log['out_image_sum'].append(energy_sum)
    log['out_image_num'].append(energy_num)
    log['out_cluster_sum'].append(cluster_sum)
    log['out_unass_sum'].append(unass_sum)


# Fill SuperaAtomic class and hand off to label-making
def run_supera(out_file='larcv.root',
               in_file='',
               config_key='',
               num_events=-1,
               num_skip=0,
               ignore_bad_association=True,
               save_log=None):

    start_time = time.time()

    writer = get_iomanager(out_file)
    driver = get_larnd2supera(config_key)
    reader = larnd2supera.reader.InputReader(driver.parser_run_config(),in_file)

    id_vv=ROOT.std.vector("std::vector<unsigned long>")()
    value_vv=ROOT.std.vector("std::vector<float>")()

    id_v=ROOT.std.vector("unsigned long")()
    value_v=ROOT.std.vector("float")()

    if num_events < 0:
        num_events = len(reader)

    print("--- startup {:.2e} seconds ---".format(time.time() - start_time))

    LOG_KEYS  = ['event_id','time_read','time_convert','time_generate', 'time_store', 'time_event']
    LOG_KEYS += ['raw_image_sum','raw_image_npx','raw_packet_sum','raw_packet_num',
    'in_cluster_sum','in_unass_sum','out_image_sum','out_image_num',
    'out_cluster_sum','out_unass_sum']

    logger = dict()
    if save_log:
        for key in LOG_KEYS:
            logger[key]=[]
        driver.log(logger)
        
    for entry in range(len(reader)):

        if num_skip and entry < num_skip:
            continue

        if num_events <= 0:
            break

        num_events -= 1 

        print(f'Processing Entry {entry}')

        t0 = time.time()
        input_data = reader.GetEntry(entry)
        is_good_event = reader.CheckIntegrity(input_data,ignore_bad_association)
        if not is_good_event:
            print('[ERROR] Skipping the entry')
            continue
        time_read = time.time() - t0
        
        t1 = time.time()
        EventInput = driver.ReadEvent(input_data)
        time_convert = time.time() - t1

        # TODO Seems to run, but how to check it's really working?
        # TODO Should this be supera_driver or driver?
        t2 = time.time()
        driver.GenerateImageMeta(EventInput)
        driver.GenerateLabel(EventInput) 
        time_generate = time.time() - t2

        # Perform an integrity check
        if save_log:
            log_supera_integrity_check(EventInput,driver,logger,verbose)

        # Start data store process
        t3 = time.time()
        result = driver.Label()
        meta   = larcv_meta(driver.Meta())
        
        tensor_energy = writer.get_data("sparse3d","pcluster")
        result.FillTensorEnergy(id_v,value_v)
        larcv.as_event_sparse3d(tensor_energy,meta,id_v,value_v)

        tensor_packets = writer.get_data("sparse3d","packets")
        driver.Meta().edep2voxelset(driver._edeps_all).fill_std_vectors(id_v,value_v)
        larcv.as_event_sparse3d(tensor_packets,meta,id_v,value_v)

        tensor_semantic = writer.get_data("sparse3d","pcluster_semantics")
        result.FillTensorSemantic(id_v,value_v)
        larcv.as_event_sparse3d(tensor_semantic,meta,id_v,value_v)

        cluster_energy = writer.get_data("cluster3d","pcluster")
        result.FillClustersEnergy(id_vv,value_vv)
        larcv.as_event_cluster3d(cluster_energy,meta,id_vv,value_vv)

        cluster_dedx = writer.get_data("cluster3d","pcluster_dedx")
        result.FillClustersdEdX(id_vv,value_vv)
        larcv.as_event_cluster3d(cluster_dedx,meta,id_vv,value_vv)
        
        particle = writer.get_data("particle","pcluster")
        for p in result._particles:
            if not p.valid:
                continue
            larp = larcv_particle(p)
            particle.append(larp)

        # TODO fill the run ID 
        writer.set_id(0,0,int(input_data.event_id))
        writer.save_entry()
        time_store = time.time() - t3

        time_event = time.time() - t0
        print("--- running driver  {:.2e} seconds ---".format(time_event))

        if save_log:
            logger['event_id'].append(input_data.event_id)
            logger['time_read'    ].append(time_read)
            logger['time_convert' ].append(time_convert)
            logger['time_generate'].append(time_generate)
            logger['time_store'   ].append(time_store)
            logger['time_event'   ].append(time_event)

    writer.finalize()

    # store supera log dictionary
    if save_log:
        np.savez('log_larnd2supera.npz',**logger)

    print("done")   








