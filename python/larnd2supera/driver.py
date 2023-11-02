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

    LOG_KEYS = ('ass_saturation',
        'residual_q',
        'packet_total',
        'packet_ctr',
        'packet_noass',
        'packet_badass',
        'fraction_nan',
        'ass_frac')

    def __init__(self):
        super().__init__()
        self._geom_dict  = None
        self._run_config = None
        self._trackid2idx = std.vector('supera::Index_t')()
        self._allowed_detectors = std.vector('std::string')()
        self._edeps_unassociated = std.vector('supera::EDep')()
        self._edeps_all = std.vector('supera::EDep')()
        self._ass_distance_limit=0.4434*4.5
        self._ass_charge_limit=0.05
        self._log=None
        self._electron_energy_threshold=0
        self._estimate_pt_time=True
        self._ignore_bad_association=True
        print("Initialized SuperaDriver class")


    def parser_run_config(self):
        return self._run_config


    def log(self,data_holder):

        for key in self.LOG_KEYS:
            if key in data_holder:
                raise KeyError(f'Key {key} exists in the log data holder already.')
            data_holder[key]=[]
        self._log = data_holder


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
                except ValueError:
                    print('Failed to load with PropertyKeyword',cfg_dict['PropertyKeyword'])
                    print('Supported types:', LarpixParser.util.configuration_keywords())
                    return False
        else:
            print('PropertyKeyword missing. Loading TileLayout and DetectorProperties...')
            if not 'TileLayout' in cfg_dict:
                print('TileLayout not in the configuration data!')
                return False

            if not 'DetectorProperties' in cfg_dict:
                print('DetectorProperties not in the configuration data!')
                raise False

            self._geom_dict  = LarpixParser.util.load_geom_dict(cfg_dict['TileLayout'])
            self._run_config = LarpixParser.util.get_run_config(cfg_dict['DetectorProperties'])

        # Event separator default value needs to be set.
        # We repurpose "run_config" of EventParser to hold this attribute.
        self._run_config['event_separator'] = 'eventID'
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
            self._estimate_pt_time = cfg.get('EstimatePointTime',
                self._estimate_pt_time)
            self._ass_distance_limit = cfg.get('AssDistanceLimit',
                self._ass_distance_limit)
            self._ass_charge_limit = cfg.get('AssChargeLimit',
                self._ass_charge_limit)

        super().ConfigureFromFile(fname)


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


    def ReadEvent(self, data, verbose=False):
        
        start_time = time.time()

        # initialize the new event record
        if not self._log is None:
            for key in self.LOG_KEYS:
                self._log[key].append(0)

        supera_event = supera.EventInput()
        supera_event.reserve(len(data.trajectories))
        
        # 1. Loop over trajectories, create one supera::ParticleInput for each
        #    store particle inputs in list to fill parent information later
        max_trackid = max(data.trajectories['trackID'].max(),data.segments['trackID'].max())
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
                if verbose:
                    print('  TrackID',part_input.part.trackid,
                          'PDG',part_input.part.pdg,
                          'Energy',part_input.part.energy_init)
            if traj['trackID'] < 0:
                print('Negative track ID found',traj['trackID'])
                raise ValueError
            self._trackid2idx[int(traj['trackID'])] = part_input.part.id
            supera_event.push_back(part_input)
            
        if verbose:
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
        if verbose:
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
        if not self._log is None:
            #if len(bad_track_ids):
            #    self._log['bad_track_id'][-1]+=1
            self._log['packet_total'][-1]+=len(data.packets)

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
        for ip, packet in enumerate(data.packets):
            if verbose:
                print('*****************Packet', ip, '**********************')

            # If packet_type !=0 continue
            if packet['packet_type'] != 0: continue

            # Record this packet
            raw_edep = supera.EDep()
            raw_edep.x, raw_edep.y, raw_edep.z = x[ip]*self._mm2cm, y[ip]*self._mm2cm, z[ip]*self._mm2cm
            raw_edep.e = dE[ip]
            self._edeps_all.push_back(raw_edep)
            check_raw_sum += dE[ip]

            # Log the valid packet count
            if not self._log is None:
                self._log['packet_ctr'][-1]+=1

            # We analyze and modify segments and fractions, so make a copy
            packet_segments  = np.array(ass_segments[ip])
            packet_fractions = np.array(ass_fractions[ip])
            packet_edeps = [None] * len(packet_segments)

            #tids = [data.segments[packet_segments[idx]]['trackID'] for idx in range(packet_segments.shape[0])]
            #debug1 = 10 in tids
            #debug2 = False
            #if debug1:
            #    d='1068 1273 1274 1290 1320 1342 1375 1425 1463 1472 1473 1479 1517 1069 1071 1078 1089 1091 1093 1098 1102 1114 1149 1171 1235 1242 1250 1251 1253 1256 1081 1084 1177 1178 1092 1099 1100 1103 1130 1179 1270 1544 1576 1587 1590 1592 1597 1610 1611 1628 1636 1639 1545 1287 1376 1288 1291 1293 1584 1588 1593 1595 1608 1151 1152 1218 1224 1556 1849'
            #    for t in [int(val) for val in d.split()]:
            #        if t in tids:
            #            debug2=True
            #            break
            #if debug1 and debug2:
            #    verbose = True
            #else:
            #    verbose = False

            # Initialize seg_flag once 
            if seg_flag is None:
                seg_flag = np.zeros(shape=(packet_segments.shape[0]),dtype=bool)
                seg_dist = np.zeros(shape=(packet_segments.shape[0]),dtype=float)
            seg_flag[:] = True
            seg_dist[:] = 1.e9

            # Step 1. Compute the distance and reject some segments (see above comments for details)
            for it in range(packet_segments.shape[0]):
                seg_idx = packet_segments[it]
                if seg_idx == -1: 
                    seg_flag[it]=False
                    continue                
                # If logging, record the sum of the raw packet fraction
                if not self._log is None:
                    self._log['ass_frac'][-1] += packet_fractions[it]
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

                seg_dist[it] = poca_pt.distance(packet_pt)
                edep.x, edep.y, edep.z = packet_pt.x, packet_pt.y, packet_pt.z
                edep.dedx = seg['dEdx']
                packet_edeps[it] = edep

            if verbose:
                print('[INFO] Assessing packet',ip)
                print('       Segments :', packet_segments)
                print('       TrackIDs :', [data.segments[packet_segments[idx]]['trackID'] for idx in range(packet_segments.shape[0])])
                print('       Fractions:', ['%.3f' % f for f in packet_fractions])
                print('       Energy   : %.3f' % dE[ip])
                print('       Position :', ['%.3f' % f for f in [x[ip]*self._mm2cm,y[ip]*self._mm2cm,z[ip]*self._mm2cm]])
                print('       Distance :', ['%.3f' % f for f in seg_dist])

            # Post-Step1, check if there is any associated segments left.
            # Post-Step1.A: If none left, then register to the set of unassociated segments.
            if (seg_dist < self._ass_distance_limit).sum()<1:
                if verbose:
                    print('[WARNING] found a packet unassociated with any tracks')
                    print('          Packet %d Charge %.3f' % (ip,dE[ip]))
                edep = supera.EDep()
                edep.x,edep.y,edep.z,edep.e = x[ip]*self._mm2cm, y[ip]*self._mm2cm, z[ip]*self._mm2cm, dE[ip]
                self._edeps_unassociated.push_back(edep)
                if not self._log is None:
                    self._log['packet_noass'][-1] += 1
                check_ana_sum += edep.e
                continue

            # After this point, >=1 association will be found. Log if needed
            if not self._log is None:
                self._log['ass_frac'][-1] += 1

            # Post-Step1.B: If only one segment left, register to that packet.
            if (seg_dist < self._ass_distance_limit).sum()==1:
                if verbose:
                    print('[INFO] Registering the only segment within distance limit',packet_segments[it])
                it = np.argmin(seg_dist)
                seg_idx = packet_segments[it]
                seg = data.segments[seg_idx]
                packet_edeps[it].dedx = seg['dEdx']
                packet_edeps[it].e = dE[ip]
                supera_event[self._trackid2idx[int(seg['trackID'])]].pcloud.push_back(packet_edeps[it])
                check_ana_sum += packet_edeps[it].e
                continue

            # Check (and log if needed) if the number of associations is saturated 
            if not -1 in packet_segments:
                if verbose:
                    print('[WARNING] found',len(packet_segments),'associated track IDs maxing out the recording array size')
                if not self._log is None:
                    self._log['ass_saturation'][-1] += 1

            # Step 2. Claculate the fraction min threshold and ignore segments with a fraction value below
            frac_min = 0
            nan_fraction_found=False
            if abs(dE[ip])>0.:
                frac_min = abs(self._ass_charge_limit / dE[ip])
            for it in range(packet_fractions.shape[0]):
                # If the associated fraction of this segment is nan, disregard this segment
                if np.isnan(packet_fractions[it]):
                    seg_flag[it]=False
                    nan_fraction_found=True
                    continue
                if seg_dist[it] > self._ass_distance_limit:
                    seg_flag[it]=False
                    continue
                seg_flag[it] = abs(packet_fractions[it]) > frac_min

            if nan_fraction_found:
                print('    WARNING: found nan in fractions of a packet:', packet_fractions)
                if self._log is not None:
                    self._log['fraction_nan'][-1] += 1


            # Check if none of segments is above the frac_min. If so, only use the shortest distant segment.
            if seg_flag.sum()<1:
                if verbose:
                    print('[WARNING] All associated segment below threshold: packet')
                    print('          Packet %d Charge %.3f' % (ip,dE[ip]), ['%.3f' % f for f in packet_fractions])
                if not self._log is None:
                    self._log['packet_badass'][-1] += 1            
                it = np.argmin(seg_dist)
                if verbose:
                    print('[INFO] Registering the closest segment',packet_segments[it])
                #print(ass_segments[ip])
                #print(ass_fractions[ip])
                #print(it)
                #print(packet_fractions)
                seg_idx = packet_segments[it]
                seg = data.segments[seg_idx]
                packet_edeps[it].dedx = seg['dEdx']
                packet_edeps[it].e = dE[ip]
                supera_event[self._trackid2idx[int(seg['trackID'])]].pcloud.push_back(packet_edeps[it])
                check_ana_sum += packet_edeps[it].e
                continue


            # Step 3: re-normalize the fraction among remaining segments and create EDeps
            frac_norm = 1. / (np.abs(packet_fractions[seg_flag]).sum())
            for it in range(packet_fractions.shape[0]):
                if not seg_flag[it]: continue
                seg_idx = packet_segments[it]
                seg = data.segments[seg_idx]
                packet_edeps[it].dedx = seg['dEdx']
                packet_edeps[it].e = dE[ip] * abs(packet_fractions[it]) * frac_norm
                supera_event[self._trackid2idx[int(seg['trackID'])]].pcloud.push_back(packet_edeps[it])
                if verbose:
                    print('[INFO] Registered segment',seg_idx)
                check_ana_sum += packet_edeps[it].e

        if verbose:
            print("--- filling edep %s seconds ---" % (time.time() - start_time)) 

        if not self._log is None:

            self._log['residual_q'][-1] = check_raw_sum - check_ana_sum

            if self._log['packet_ctr'][-1]>0:
                self._log['ass_frac'][-1] /= self._log['packet_ctr'][-1]

            if self._log['packet_noass'][-1]:
                value_bad, value_frac = self._log['packet_noass'][-1], self._log['packet_noass'][-1]/self._log['packet_total'][-1]*100.
                print(f'    WARNING: {value_bad} packets ({value_frac} %) had no MC track association')

            if self._log['fraction_nan'][-1]:
                value_bad, value_frac = self._log['fraction_nan'][-1], self._log['fraction_nan'][-1]/self._log['packet_total'][-1]*100.
                print(f'    WARNING: {value_bad} packets ({value_frac} %) had nan fractions associated')

            if self._log['ass_frac'][-1]<0.9999:
                print(f'    WARNING associated packet count fraction is {self._log["ass_frac"][-1]} (<1.0)')

            #if self._log['bad_track_id'][-1]:
            #    print(f'    WARNING: {self._log["bad_track_id"][-1]} invalid track IDs found in the association')

        if abs(check_raw_sum - check_ana_sum)>0.1:
            print('[WARNING] large disagreement in the sum packet values:')
            print('    Raw sum:',check_raw_sum)
            print('    Accounted sum:',check_ana_sum)
        supera_event.unassociated_edeps = self._edeps_unassociated
        return supera_event

    def TrajectoryToParticle(self, trajectory):
        p = supera.Particle()
        # Larnd-sim stores a lot of these fields as numpy.uint32, 
        # but Supera/LArCV want a regular int, hence the type casting
        # TODO Is there a cleaner way to handle this?
        p.id             = int(trajectory['eventID'])
        #p.interaction_id = trajectory['interactionID']
        p.trackid        = int(trajectory['trackID'])
        p.pdg            = int(trajectory['pdgId'])
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

        traj_parent_id = trajectory['parentID']
        # This now causes errors?
        #if traj_parent_id == -1: p.parent_trackid = supera.kINVALID_TRACKID
        if traj_parent_id == -1: p.parent_trackid = p.trackid
        else:                    p.parent_trackid = int(trajectory['parentID'])

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
        
        if pdg_code == 2112 or pdg_code > 1000000000:
            supera_part.type = supera.kNeutron
        
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
                              "TrackId",edepsim_part['trackID'],
                              "Kinetic Energy",ke,
                              "Parent PDG",supera_part.parent_pdg ,
                              "Parent TrackId",edepsim_part['parentID'],
                              "G4ProcessType",g4type_main ,
                              "SubProcessType",g4type_sub)
                        supera_part.type = supera.kIonization
                #elif g4type_sub == 151:

                else:
                    print("    WARNING: UNEXPECTED EM SubType ")
                    print("      PDG",pdg_code,
                          "TrackId",edepsim_part['trackID'],
                          "Kinetic Energy",ke,
                          "Parent PDG",supera_part.parent_pdg ,
                          "Parent TrackId",edepsim_part['parentID'],
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
                      "TrackId",edepsim_part['trackID'],
                      "Kinetic Energy",ke,
                      "Parent PDG",supera_part.parent_pdg ,
                      "Parent TrackId",edepsim_part['parentID'],
                      "G4ProcessType",g4type_main ,
                      "SubProcessType",g4type_sub)

                if ke < self._electron_energy_threshold:
                    supera_part.type = supera.kCompton
                else:
                    supera_part.type = supera.kOtherShower
        else:
            supera_part.type = supera.kTrack
