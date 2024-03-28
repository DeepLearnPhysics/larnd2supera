import time
import edep2supera, ROOT
from ROOT import supera,std,TG4TrajectoryPoint
import numpy as np
import LarpixParser
from LarpixParser import hit_parser as HitParser
import yaml
from yaml import Loader
import larnd2supera



class SuperaDriver(edep2supera.edep2supera.SuperaDriver):

    LOG_KEYS = ('ass_saturation', # number of packets where the association array is full (target 0)
        'residual_q',             # anaccounted energy summed over all packets (target 0)
        'packet_ctr',             # the total number of "data" (type==0) packets (arbitrary, N)
        'packet_noass_input',     # number of packets with no associated segments from the input (target 0)
        'packet_noass',           # number of packets with no valid association could be made (target 0)
        'packet_frac_sum',        # the average sum of fraction values from the input
        'fraction_nan',           # number of packets that contain >=1 nan in associated fraction (target 0)
        'ass_frac',               # the fraction of total packets where >=1 association could be made (target 1)
        'ass_charge_frac',        # the average of total charge fraction used from the input fraction info (target 1)
        'ass_drop_charge',        # the sum of fraction of associations dropped by charge (target 0)
        'ass_negative_charge',    # the sum of fraction of negative association droped by charge (arbitrary)
        'ass_drop_dist',          # the sum of fraction of associations dropped by the distance (target 0)
        'drop_ctr_total',
        'drop_ctr_dist3d',
        'drop_ctr_drift_dist',
        'drop_ctr_low_charge',
        'drop_ctr_negative_charge',
        )

    def __init__(self):
        super().__init__()
        self._geom_dict  = None
        self._run_config = None
        self._trackid2idx = std.vector('supera::Index_t')()
        self._allowed_detectors = std.vector('std::string')()
        self._edeps_unassociated = std.vector('supera::EDep')()
        self._edeps_all = std.vector('supera::EDep')()
        self._ass_distance_limit=0.4434*6
        self._ass_charge_limit=0.00
        self._ass_time_future=20
        self._ass_time_past=5 # 1.8
        self._log=None
        self._electron_energy_threshold=0
        self._search_association=True
        print("Initialized SuperaDriver class")


    def parser_run_config(self):
        return self._run_config


    def log(self,data_holder):

        for key in self.LOG_KEYS:
            if key in data_holder:
                raise KeyError(f'Key {key} exists in the log data holder already.')
            data_holder[key]=[]
        self._log = data_holder

    def drift_dir(self,xyz):

        from larndsim.consts import detector
            
        pixel_plane = -1
        for ip, plane in enumerate(detector.TPC_BORDERS):
            if not plane[0][0]-2e-2 <= xyz[2] <= plane[0][1]+2e-2: continue
            if not plane[1][0]-2e-2 <= xyz[1] <= plane[1][1]+2e-2: continue
            if not min(plane[2][1]-2e-2,plane[2][0]-2e-2) <= xyz[0] <= max(plane[2][1]+2e-2,plane[2][0]+2e-2):
                continue
            pixel_plane=ip
            break
                    
        if pixel_plane < 0:
            #raise ValueError(f'Could not find pixel plane id for xyz {xyz}')
            return 0

        edges = detector.TPC_BORDERS[pixel_plane][2]
        if edges[1] > edges[0]:
            return -1
        else:
            return 1

    def associated_along_drift(self, seg, packet_pt, raise_error=True, verbose=False):

        from larndsim.consts import detector
        # project on 2D, find the closest point on YZ plane 
        a = np.array([seg['x_start'],seg['y_start'],seg['z_start']],dtype=float)
        b = np.array([seg['x_end'],seg['y_end'],seg['z_end']],dtype=float)
        frac = self.PoCA_numpy(a[1:],b[1:],packet_pt[1:],scalar=True)
        # infer the 3d point
        seg_pt = a + frac*(b-a)

        # Check the drift direction
        directions = [self.drift_dir(pt) for pt in [a,b,seg_pt]]
        if 1 in directions and -1 in directions:
            if raise_error:
                #print(f'start {a}\nend {b}\nyz {seg_pt}\npacket {packet_pt}')
                raise RuntimeError(f'Found a packet with ambiguous drift direction start/end/yz {directions}')
            return False
        elif -1 in directions:
            # signal | segment | induced signal
            low = seg_pt[0] - self._ass_time_future * detector.V_DRIFT
            hi  = seg_pt[0] + self._ass_time_past * detector.V_DRIFT
            #return low < packet_pt[0] < hi
        elif 1 in directions:
            # induced signal | segment | signal
            low = seg_pt[0] - self._ass_time_past * detector.V_DRIFT
            hi  = seg_pt[0] + self._ass_time_future * detector.V_DRIFT
            #return low < packet_pt[0] < hi
        elif raise_error:
            print(f'start {a}\nend {b}\nyz {seg_pt}\npacket {packet_pt}')
            raise RuntimeError(f'Found a packet with ambiguous drift direction start/end/yz {directions}')
        else:
            return False


        if verbose:
            print('Along-drift dist check:',low,'<',packet_pt[0],'<',hi)
        return low < packet_pt[0] < hi

    def LoadPropertyConfigs(self,cfg_dict):

        # Expect only PropertyKeyword or (TileLayout,DetectorProperties). Not both.
        if cfg_dict.get('PropertyKeyword',None):
            if cfg_dict.get('TileLayout',None) or cfg_dict.get('DetectorProperties',None):
                print('PropertyKeyword provided:', cfg_dict['PropertyKeyword'])
                print('But also founnd below:')
                for keyword in ['TileLayout','DetectorProperties']:
                    print('%s: "%s"' % (keyword,cfg_dict.get(keyword,None)))
                    print('Bool',bool(cfg_dict.get(keyword,None)))

                print('You cannot specify duplicated property infomration!')
                return False
            else:
                try:
                    self._run_config, self._geom_dict = LarpixParser.util.detector_configuration(cfg_dict['PropertyKeyword'])
                    from larndsim.consts import detector
                    detector.load_detector_properties(cfg_dict['PropertyKeyword'])

                except ValueError:
                    print('Failed to load with PropertyKeyword',cfg_dict['PropertyKeyword'])
                    print('Supported types:', LarpixParser.util.configuration_keywords())
                    return False
        else:
            raise RuntimeError('Must contain "PropertyKeyword". Currently other run modes not supported.')

        # Event separator default value needs to be set.
        # We repurpose "run_config" of EventParser to hold this attribute.
        self._run_config['event_separator'] = 'event_id'
        # Apply run config modification if requested
        run_config_mod = cfg_dict.get('ParserRunConfig',None)
        if run_config_mod:
            for key,val in run_config_mod.items():
                self._run_config[key]=val
        return True


    def ConfigureFromFile(self,fname):
        with open(fname,'r') as f:
            cfg=yaml.load(f.read(),Loader=Loader)
            if not self.LoadPropertyConfigs(cfg):
                raise ValueError('Failed to configure larnd2supera!')
            self._electron_energy_threshold = cfg.get('ElectronEnergyThreshold',
                self._electron_energy_threshold
                )
            self._ass_distance_limit = cfg.get('AssDistanceLimit',
                self._ass_distance_limit)
            self._ass_charge_limit = cfg.get('AssChargeLimit',
                self._ass_charge_limit)
            self._search_association = cfg.get('SearchAssociation',
                self._search_association)
        super().ConfigureFromFile(fname)


    def PoCA_numpy(self, a, b, pt, scalar=False):

        ab = b - a

        t = (pt - a).dot(ab)

        if t <= 0.: 
            return 0. if scalar else a
        else:
            denom = ab.dot(ab)
            if t >= denom:
                return 1. if scalar else b
            else:
                return t/denom if scalar else a + ab.dot(t/denom)

    def PoCA(self, a, b, pt, scalar=False):
        
        ab = b - a
        
        t = (pt - a) * ab
        
        if t <= 0.: 
            return 0. if scalar else a
        else:
            denom = ab * ab
            if t >= denom:
                return 1. if scalar else b
            else:
                return t/denom if scalar else a + ab * t/denom

    def ConfigureFromText(self,txt):
        cfg=yaml.load(txt,Loader=Loader)
        if not self.LoadPropertyConfigs(cfg):
            raise ValueError('Failed to configure larnd2supera!')
            self._electron_energy_threshold = cfg.get('ElectronEnergyThreshold',
                self._electron_energy_threshold
                )
        super().ConfigureFromText(txt)


    def ReadEvent(self, data, verbose=0):
        
        start_time = time.time()

        # initialize the new event record
        if not self._log is None:
            for key in self.LOG_KEYS:
                self._log[key].append(0)

        supera_event = supera.EventInput()
        supera_event.reserve(len(data.trajectories))
        
        # 1. Loop over trajectories, create one supera::ParticleInput for each
        #    store particle inputs in list to fill parent information later
        max_trackid = max(data.trajectories['traj_id'].max(),data.segments['traj_id'].max())
        self._trackid2idx.resize(int(max_trackid+1),supera.kINVALID_INDEX)
        for traj in data.trajectories:
            # print("traj",traj)
            part_input = supera.ParticleInput()

            part_input.valid = True
            part_input.part  = self.TrajectoryToParticle(traj)
            part_input.part.id = supera_event.size()
            #part_input.type  = self.GetG4CreationProcess(traj)
            #supera_event.append(part_input)
            if self.GetLogger().verbose():
                if verbose > 1:
                    print('  TrackID',part_input.part.trackid,
                          'PDG',part_input.part.pdg,
                          'Energy',part_input.part.energy_init)
            if traj['traj_id'] < 0:
                print('Negative track ID found',traj['traj_id'])
                raise ValueError
            self._trackid2idx[int(traj['traj_id'])] = part_input.part.id
            supera_event.push_back(part_input)
            
        if verbose > 0:
            print("--- trajectory filling %s seconds ---" % (time.time() - start_time)) 
        start_time = time.time()  

        # 2. Fill parent information for ParticleInputs created in previous loop
        for i,part in enumerate(supera_event):
            traj = data.trajectories[i]

            parent=None
            if(part.part.parent_trackid < self._trackid2idx.size()):
                parent_index = self._trackid2idx[part.part.parent_trackid]
                if not parent_index == supera.kINVALID_INDEX:
                    parent = supera_event[parent_index].part
                    part.part.parent_pdg = parent.pdg
                    
            self.SetProcessType(traj,part.part,parent)

        # 3. Loop over "voxels" (aka packets), get EDep from xyz and charge information,
        #    and store in pcloud
        x, y, z, dE = HitParser.hit_parser_energy(data.t0, data.packets, self._geom_dict, self._run_config, switch_xz=True)
        if verbose>1:
            print('Got x,y,z,dE = ', x, y, z, dE)

        start_time = time.time()

        ass_segments = data.mc_packets_assn['track_ids']
        ass_fractions = data.mc_packets_assn['fraction']
        
        # Check if the data.first_track_id is consistent with track_ids.
        # Commented out on May 15 2023 - the below code is redundant with np.nonzero in reader.py
        #bad_track_ids = [ v for v in np.unique(track_ids[track_ids<data.first_track_id]) if not v == -1]
        #if len(bad_track_ids):
        #    print(f'\n[ERROR] unexpected track ID found {bad_track_ids} (first track ID in the event {data.first_track_id})')
        #    for bad_tid in bad_track_ids:
        #        print(f'    Track ID {bad_tid} ... associated fraction {ass_fractions[np.where(track_ids==bad_tid)]}')

        ass_segments = np.subtract(ass_segments, data.segment_index_min*(ass_segments!=-1))

        # Define some objects that are repeatedly used within the loop
        seg_pt0   = supera.Point3D()
        seg_pt1   = supera.Point3D()
        packet_pt = supera.Point3D()
        poca_pt   = supera.Point3D()
        seg_flag  = None
        seg_dist  = None

        # a list to keep energy depositions w/o true association
        self._edeps_unassociated.clear() 
        self._edeps_unassociated.reserve(len(data.packets))
        self._edeps_all.clear();
        self._edeps_all.reserve(len(data.packets))
        self._mm2cm = 0.1 # For converting packet x,y,z values

        #
        # Loop over packets and decide particle trajectory segments that are associated with it.
        # Also calculate how much fraction of the packet value should be associated to this particle.
        #
        # Loop over packets, and for each packet:
        #   Step 1. Loop over associated segments and calculate how far the packet location (3D point)
        #           is with respect to the truth segment (3D line). If the distance is larger than the
        #           parameter self._ass_distance_limit, ignore this segment from the association. If 
        #           this step results in no associated segment to this packet, assign the packet as
        #           self._edeps_unassociated container and continue.
        #
        #   Step 2. Calculate threshold on the associated fraction by f_min = self._ass_charge_limit/Q
        #           where Q is the total charge stored in the packet. Ignore the associated segments if its
        #           fraction is smaller than f_min. If this step results in no associated segment to this
        #           packet, associate the entire fraction to the segment with the largest fraction.
        #
        #   Step 3. If there is more than one segment remained with association to this packet, re-normalize
        #           the fraction array within the remaining elements and compute the associated charge fraction
        #           to each segment, store EDeps to corresponding particle object.
        #
        check_raw_sum=0
        check_ana_sum=0
        print('Looping over packets')
        for ip, packet in enumerate(data.packets):

            if verbose > 1:
                print('*****************Packet', ip, '**********************')

            # If packet_type !=0 continue
            if packet['packet_type'] != 0: continue

            # Record this packet
            raw_edep = supera.EDep()
            raw_edep.x, raw_edep.y, raw_edep.z = x[ip]*self._mm2cm, y[ip]*self._mm2cm, z[ip]*self._mm2cm
            raw_edep.e = dE[ip]
            self._edeps_all.push_back(raw_edep)
            check_raw_sum += dE[ip]

            # We analyze and modify segments and fractions, so make a copy
            packet_segments  = np.array(ass_segments[ip])
            packet_fractions = np.array(ass_fractions[ip])
            packet_edeps = [None] * len(packet_segments)


            for seg_id in packet_segments:
                seg=data.segments[seg_id]

            #
            # Check packet segments quality
            # - any association?
            # - Saturated?
            # - nan?
            if (packet_segments == -1).sum() == len(packet_segments):
                if verbose > 0:
                    print('[WARNING] found a packet with no association!')
                if not self._log is None:
                    self._log['packet_noass_input'][-1] += 1
            if not -1 in packet_segments:
                if verbose > 1:
                    print('[INFO] found',len(packet_segments),'associated track IDs maxing out the recording array size')
                if not self._log is None:
                    self._log['ass_saturation'][-1] += 1
            if np.isnan(packet_segments).sum() > 0:
                print('    [ERROR]: found nan in fractions of a packet:', packet_fractions)
                if not self._log is None:
                    self._log['fraction_nan'][-1] += 1


            # Initialize seg_flag once 
            if seg_flag is None:
                seg_flag = np.zeros(len(packet_segments),bool)
                #seg_dist = np.zeros(shape=(packet_segments.shape[0]),dtype=float)
            seg_flag[:] = ~(packet_segments < 0)
            #seg_dist[:] = 1.e9
            if not self._log is None:
                self._log['packet_frac_sum'][-1] += packet_fractions[seg_flag].sum()

            # Ignore packets...
            # 1. with too small fraction (in relative and absolute)
            # 2. with associated segments with invalid track id
            xyz = np.array([raw_edep.x,raw_edep.y,raw_edep.z])
            for it,f in enumerate(packet_fractions):
                if not seg_flag[it]:
                    continue

                if f <= 0.:
                    seg_flag[it] = False
                    if not self._log is None:
                        self._log['ass_negative_charge'][-1] += f * dE[ip]
                        self._log['drop_ctr_negative_charge'][-1] += 1
                    continue

                if f * dE[ip] < self._ass_charge_limit:
                    seg_flag[it] = False
                    if not self._log is None:
                        self._log['ass_drop_charge'][-1] += packet_fractions[it]
                        self._log['drop_ctr_low_charge'][-1] +=1
                    continue

                # check if an associated trajectory ID is invalid
                seg = data.segments[packet_segments[it]]
                traj_id = int(seg['traj_id'])
                if traj_id >= self._trackid2idx.size() or self._trackid2idx[traj_id]==supera.kINVALID_INDEX:
                    print(f'[ERROR] found a segment with the invalid traj_id={traj_id}')
                    seg_flag[it] = False
                    continue

                # Check if the segment should be associated along the drift
                if not self.associated_along_drift(seg,xyz):
                    seg_flag[it] = False
                    if not self._log is None:
                        self._log['drop_ctr_drift_dist'][-1] += 1
                    continue

                seg_flag[it] = True

            # Step 1. Compute the distance and reject some segments (see above comments for details)'
            for it in range(packet_segments.shape[0]):
                if not seg_flag[it]:
                    continue
                seg_idx = packet_segments[it]
                # Access the segment
                seg = data.segments[seg_idx]
                # Compute the Point of Closest Approach as well as estimation of time.
                edep = supera.EDep()
                seg_pt0.x, seg_pt0.y, seg_pt0.z = seg['x_start'], seg['y_start'], seg['z_start']
                seg_pt1.x, seg_pt1.y, seg_pt1.z = seg['x_end'], seg['y_end'], seg['z_end']
                packet_pt.x, packet_pt.y, packet_pt.z = x[ip]*self._mm2cm, y[ip]*self._mm2cm, z[ip]*self._mm2cm

                if seg['t0_start'] < seg['t0_end']:
                    time_frac = self.PoCA(seg_pt0,seg_pt1,packet_pt,scalar=True)
                    edep.t = seg['t0_start'] + time_frac * (seg['t0_end'  ] - seg['t0_start'])
                    poca_pt = seg_pt0 + (seg_pt1 - seg_pt0) * time_frac

                else:
                    time_frac = self.PoCA(seg_pt1,seg_pt0,packet_pt,scalar=True)
                    edep.t = seg['t0_end'  ] + time_frac * (seg['t0_start'] - seg['t0_end'  ])
                    poca_pt = seg_pt1 + (seg_pt0 - seg_pt1) * time_frac

                #seg_dist[it] = poca_pt.distance(packet_pt)
                seg_dist = poca_pt.distance(packet_pt)
                if seg_dist > self._ass_distance_limit:
                    seg_flag[it] = False
                    if not self._log is None:
                        self._log['ass_drop_dist'][-1] += packet_fractions[it]
                        self._log['drop_ctr_dist3d'][-1] += 1
                    continue
                edep.x, edep.y, edep.z = packet_pt.x, packet_pt.y, packet_pt.z
                edep.dedx = seg['dEdx']
                packet_edeps[it] = edep


            # split the energy among valid, associated packets
            if seg_flag.sum() < 1:
                # no valid association
                edep = supera.EDep()
                edep.x,edep.y,edep.z,edep.e = x[ip]*self._mm2cm, y[ip]*self._mm2cm, z[ip]*self._mm2cm, dE[ip]
                self._edeps_unassociated.push_back(edep)
                check_ana_sum += edep.e
                self._log['drop_ctr_total'][-1] += 1

            else:

                # Re-compute the fractions
                fsum=packet_fractions[seg_flag].sum()
                packet_fractions[~seg_flag] = 0. 
                if fsum>0:
                    packet_fractions[seg_flag] /= fsum
                else:
                    packet_fractions[seg_flag] /= seg_flag.sum()
                for idx in np.where(seg_flag)[0]:
                    seg_idx = packet_segments[idx]
                    seg = data.segments[seg_idx]
                    packet_edeps[idx].e = dE[ip] * packet_fractions[idx]
                    #print(seg['traj_id'])
                    #print(int(seg['traj_id']))
                    #print(self._trackid2idx[int(seg['traj_id'])])
                    traj_id = int(seg['traj_id'])
                    #if traj_id < 1:
                    #    print(seg_flag)
                    #    print(packet_fractions)
                    #    print(traj_id)
                    #    print(self._trackid2idx[traj_id])
                    supera_event[self._trackid2idx[int(seg['traj_id'])]].pcloud.push_back(packet_edeps[idx])
                    check_ana_sum += packet_edeps[idx].e
                if not self._log is None:
                    self._log['ass_charge_frac'][-1] += fsum
                    self._log['ass_frac'][-1] += 1

            if verbose > 1:
                print('[INFO] Assessing packet',ip)
                print('       Associated?',seg_flag.sum())
                print('       Segments :', packet_segments)
                print('       TrackIDs :', [data.segments[packet_segments[idx]]['traj_id'] for idx in range(packet_segments.shape[0])])
                print('       Fractions:', ['%.3f' % f for f in packet_fractions])
                print('       Energy   : %.3f' % dE[ip])
                print('       Position :', ['%.3f' % f for f in [x[ip]*self._mm2cm,y[ip]*self._mm2cm,z[ip]*self._mm2cm]])

        if verbose:
            print("--- filling edep %s seconds ---" % (time.time() - start_time))

        print('Unassociated edeps',self._edeps_unassociated.size())
        if self._search_association:
            import tqdm
            # Attempt to associate unassociated edeps
            failed_unass = std.vector('supera::EDep')()
            for iedep, edep in tqdm.tqdm(enumerate(self._edeps_unassociated)):
                #print('Searching for EDep',iedep,'/',self._edeps_unassociated.size())
                ass_found=False
                for seg in data.segments:

                    xyz = np.array([edep.x,edep.y,edep.z])
                    if not self.associated_along_drift(seg,xyz,False):
                        continue

                    seg_pt0.x, seg_pt0.y, seg_pt0.z = seg['x_start'], seg['y_start'], seg['z_start']
                    seg_pt1.x, seg_pt1.y, seg_pt1.z = seg['x_end'], seg['y_end'], seg['z_end']
                    packet_pt.x, packet_pt.y, packet_pt.z = edep.x, edep.y, edep.z

                    edep_time = 0 
                    if seg['t0_start'] < seg['t0_end']:
                        time_frac = self.PoCA(seg_pt0,seg_pt1,packet_pt,scalar=True)
                        edep_time = seg['t0_start'] + time_frac * (seg['t0_end'  ] - seg['t0_start'])
                        poca_pt = seg_pt0 + (seg_pt1 - seg_pt0) * time_frac
                    else:
                        time_frac = self.PoCA(seg_pt1,seg_pt0,packet_pt,scalar=True)
                        edep_time = seg['t0_end'  ] + time_frac * (seg['t0_start'] - seg['t0_end'  ])
                        poca_pt = seg_pt1 + (seg_pt0 - seg_pt1) * time_frac

                    seg_dist = poca_pt.distance(packet_pt)
                    if seg_dist < self._ass_distance_limit:
                        #associate
                        edep.dedx = seg['dEdx']
                        edep.t    = edep_time
                        supera_event[self._trackid2idx[int(seg['traj_id'])]].pcloud.push_back(edep)

                        ass_found=True
                        break
                if not ass_found:
                    failed_unass.push_back(edep)
                    print(f'Found unassociated edep ({iedep}th) ... Energy={edep.e}')

                #print('    Found so far:',failed_unass.size())
            print('Clearing self._edeps_unassociated of size',self._edeps_unassociated.size())
            self._edeps_unassociated.clear()
            print('Size after clearing:',self._edeps_unassociated.size())
            self._edeps_unassociated = failed_unass
            print('New self._edeps_unassociated size',self._edeps_unassociated.size())

        if not self._log is None:
            self._log['packet_noass'][-1] = self._edeps_unassociated.size()
            self._log['packet_ctr'][-1]   = (data.packets['packet_type'] == 0).sum()

        print('Unassociated edeps',self._edeps_unassociated.size())

        if not self._log is None:

            self._log['residual_q'][-1] = check_raw_sum - check_ana_sum

            if self._log['packet_ctr'][-1]>0:
                self._log['ass_frac'][-1]        /= self._log['packet_ctr'][-1]
                self._log['ass_charge_frac'][-1] /= self._log['packet_ctr'][-1]
                self._log['packet_frac_sum'][-1] /= self._log['packet_ctr'][-1]
                self._log['ass_drop_charge'][-1] /= self._log['packet_ctr'][-1]
                self._log['ass_drop_dist'][-1]   /= self._log['packet_ctr'][-1]

            if self._log['packet_noass'][-1]:
                value_bad, value_frac = self._log['packet_noass'][-1], self._log['packet_noass'][-1]/self._log['packet_ctr'][-1]*100.
                print(f'    [WARNING]: {value_bad} packets ({value_frac} %) had no MC track association')

            if self._log['fraction_nan'][-1]:
                value_bad, value_frac = self._log['fraction_nan'][-1], self._log['fraction_nan'][-1]/self._log['packet_ctr'][-1]*100.
                print(f'    [WARNING]: {value_bad} packets ({value_frac} %) had nan fractions associated')

            if self._log['packet_frac_sum'][-1]<0.9999:
                print(f'    [WARNING] some input packets have the fraction sum < 1.0 (average over packets {self._log["packet_frac_sum"]})')

            if self._log['ass_frac'][-1]<0.9999:
                print(f'    [WARNING] associated packet count fraction to the total is {self._log["ass_frac"][-1]} (<1.0)')

            if self._log['ass_charge_frac'][-1]<0.9999:
                print(f'    [WARNING] the average of summed fractions after charge/dist cut {self._log["ass_charge_frac"]}')

            if self._log['ass_drop_charge'][-1]>0.0001:
                print(f'    [WARNING] the average of associated fraction dropped due to charge cut: {self._log["ass_drop_charge"]}')

            if self._log['ass_drop_dist'][-1]>0.0001:
                print(f'    [WARNING] the average of associated fraction dropped due to distance cut: {self._log["ass_drop_dist"]}')

            if self._log['drop_ctr_total'][-1]:
                for key in self._log.keys():
                    if not str(key).startswith('drop_ctr'):
                        continue
                    print(key,self._log[key])


            #if self._log['bad_track_id'][-1]:
            #    print(f'    WARNING: {self._log["bad_track_id"][-1]} invalid track IDs found in the association')

        ##if abs(check_raw_sum - check_ana_sum)>0.1:
        ##    print('[WARNING] large disagreement in the sum packet values:')
        ##    print('    Raw sum:',check_raw_sum)
        ##    print('    Accounted sum:',check_ana_sum)
        supera_event.unassociated_edeps = self._edeps_unassociated
        return supera_event

    def TrajectoryToParticle(self, trajectory):
        p = supera.Particle()
        # Larnd-sim stores a lot of these fields as numpy.uint32, 
        # but Supera/LArCV want a regular int, hence the type casting
        # TODO Is there a cleaner way to handle this?
        p.id             = int(trajectory['event_id'])
        p.interaction_id = int(trajectory['vertex_id'])   # this is the VertexID assigned at edep-sim stage
        p.trackid        = int(trajectory['traj_id'])
        p.pdg            = int(trajectory['pdg_id'])
        p.px = trajectory['pxyz_start'][0] 
        p.py = trajectory['pxyz_start'][1] 
        p.pz = trajectory['pxyz_start'][2]
        p.energy_init = np.sqrt(pow(larnd2supera.pdg2mass.pdg2mass(p.pdg),2)+pow(p.px,2)+pow(p.py,2)+pow(p.pz,2))
        p.vtx    = supera.Vertex(trajectory['xyz_start'][0], 
                                 trajectory['xyz_start'][1], 
                                 trajectory['xyz_start'][2], 
                                 trajectory['t_start']
        )
        p.end_pt = supera.Vertex(trajectory['xyz_end'][0], 
                                 trajectory['xyz_end'][1],
                                 trajectory['xyz_end'][2], 
                                 trajectory['t_end']
        )

        traj_parent_id = trajectory['parent_id']
        # This now causes errors?
        #if traj_parent_id == -1: p.parent_trackid = supera.kINVALID_TRACKID
        if traj_parent_id == -1: p.parent_trackid = p.trackid
        else:                    p.parent_trackid = int(trajectory['parent_id'])

        if supera.kINVALID_TRACKID in [p.trackid, p.parent_trackid]:
            print('Unexpected to have an invalid track ID',p.trackid,
                  'or parent track ID',p.parent_trackid)
            raise ValueError
        
        return p
        
        
    def SetProcessType(self, edepsim_part, supera_part, supera_parent):
        pdg_code    = supera_part.pdg
        g4type_main = edepsim_part['start_process']
        g4type_sub  = edepsim_part['start_subprocess']
        
        supera_part.process = '%d::%d' % (int(g4type_main),int(g4type_sub))

        ke = np.sqrt(pow(supera_part.px,2)+pow(supera_part.py,2)+pow(supera_part.pz,2))

        dr = 1.e20
        if supera_parent:
            dx = (supera_parent.end_pt.pos.x - supera_part.vtx.pos.x)
            dy = (supera_parent.end_pt.pos.y - supera_part.vtx.pos.y)
            dz = (supera_parent.end_pt.pos.z - supera_part.vtx.pos.z)
            dr = dx + dy + dz
        
        if pdg_code == 2112:
            supera_part.type = supera.kNeutron

        elif pdg_code > 1000000000:
            supera_part.type = supera.kNucleus
        
        elif supera_part.trackid == supera_part.parent_trackid:
            supera_part.type = supera.kPrimary
            
        elif pdg_code == 22:
            supera_part.type = supera.kPhoton
        
        elif abs(pdg_code) == 11:
            
            if g4type_main == TG4TrajectoryPoint.G4ProcessType.kProcessElectromagetic:
                
                if g4type_sub == TG4TrajectoryPoint.G4ProcessSubtype.kSubtypeEMPhotoelectric:
                    supera_part.type = supera.kPhotoElectron
                    
                elif g4type_sub == TG4TrajectoryPoint.G4ProcessSubtype.kSubtypeEMComptonScattering:
                    supera_part.type = supera.kCompton
                
                elif g4type_sub == TG4TrajectoryPoint.G4ProcessSubtype.kSubtypeEMGammaConversion or \
                     g4type_sub == TG4TrajectoryPoint.G4ProcessSubtype.kSubtypeEMPairProdByCharged:
                    supera_part.type = supera.kConversion
                    
                elif g4type_sub == TG4TrajectoryPoint.G4ProcessSubtype.kSubtypeEMIonization:
                    
                    if abs(supera_part.parent_pdg) == 11:
                        supera_part.type = supera.kIonization
                        
                    elif abs(supera_part.parent_pdg) in [211,13,2212,321]:
                        supera_part.type = supera.kDelta

                    elif supera_part.parent_pdg == 22:
                        supera_part.type = supera.kCompton

                    else:
                        print("    WARNING: UNEXPECTED CASE for IONIZATION ")
                        print("      PDG",pdg_code,
                              "TrackId",edepsim_part['traj_id'],
                              "Kinetic Energy",ke,
                              "Parent PDG",supera_part.parent_pdg ,
                              "Parent TrackId",edepsim_part['parent_id'],
                              "G4ProcessType",g4type_main ,
                              "SubProcessType",g4type_sub)
                        supera_part.type = supera.kIonization
                #elif g4type_sub == 151:

                else:
                    print("    WARNING: UNEXPECTED EM SubType ")
                    print("      PDG",pdg_code,
                          "TrackId",edepsim_part['traj_id'],
                          "Kinetic Energy",ke,
                          "Parent PDG",supera_part.parent_pdg ,
                          "Parent TrackId",edepsim_part['parent_id'],
                          "G4ProcessType",g4type_main ,
                          "SubProcessType",g4type_sub)
                    raise ValueError
                    
            elif g4type_main == TG4TrajectoryPoint.G4ProcessType.kProcessDecay:
                #print("    WARNING: DECAY ")
                #print("      PDG",pdg_code,
                #      "TrackId",edepsim_part['trackID'],
                #      "Kinetic Energy",ke,
                #      "Parent PDG",supera_part.parent_pdg ,
                #      "Parent TrackId",edepsim_part['parentID'],
                #      "G4ProcessType",g4type_main ,
                #      "SubProcessType",g4type_sub)
                supera_part.type = supera.kDecay

            elif g4type_main == TG4TrajectoryPoint.G4ProcessType.kProcessHadronic and g4type_sub == 151 and dr<0.0001:
                if ke < self._electron_energy_threshold:
                    supera_part.type = supera.kIonization
                else:
                    supera_part.type = supera.kDecay
            
            else:
                print("    WARNING: Guessing the shower type as", "Compton" if ke < self._electron_energy_threshold else "OtherShower")
                print("      PDG",pdg_code,
                      "TrackId",edepsim_part['traj_id'],
                      "Kinetic Energy",ke,
                      "Parent PDG",supera_part.parent_pdg ,
                      "Parent TrackId",edepsim_part['parent_id'],
                      "G4ProcessType",g4type_main ,
                      "SubProcessType",g4type_sub)

                if ke < self._electron_energy_threshold:
                    supera_part.type = supera.kCompton
                else:
                    supera_part.type = supera.kOtherShower
        else:
            supera_part.type = supera.kTrack
