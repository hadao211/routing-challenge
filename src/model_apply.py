from os import path
import json, time, numpy as np, pandas as pd, datetime, copy, re, traceback, multiprocessing as mp
from scipy.spatial import distance







# function for solving instance
def meta_zone(zone):
    m = re.match('[A-Z]+(?=-)', zone)
    higher_meta = m.group(0) if m else zone

    m_ = re.match('[A-Z]-\d+(?=.)', zone)
    meta = m_.group(0) if m_ else zone

    return meta, higher_meta  

# function for solving instance
def solve(k,route, travel_time, package, timer_start, penalty, sz_weight, last_zone_weight, max_allowed_dist):
    print("start "+str(k))
    try:
        n = len(travel_time)
        points = np.zeros((n,2)) # coords of nodes
        stops = route["stops"] # stop dictionary
        nodes = [k for k,v in travel_time.items()] # node ids of nodes
        zones_l = [stops[k]["zone_id"] for k,v in travel_time.items()] # zone ids of nodes
        types = [stops[k]["type"] for k,v in travel_time.items()] # stop types of nodes
        
        tt = np.zeros((n+1,n+1)) # convert travel matrix from dict to np array
        # add artificial node n+1 as end node of last path
        tt[n, :] = 999999 
        tt[:, n] = 0
        tt[n,n] = 999999
        
        # fill tt and node coords
        for i in range(n):
            tt[i,0:n] = list(travel_time[nodes[i]].values())
            tt[i,i] = 999999
            points[i,0] = stops[nodes[i]]["lat"]
            points[i,1] = stops[nodes[i]]["lng"]
            
        # impute missing zone data ###################################################################
        nans = [i for i in range(len(zones_l)) if pd.isna(zones_l[i])]
        
        for i in nans: 
            if types[i] == "Station":
                zones_l[i] = "Start"
            else:
                closest = min([tt[i,j] for j in range(n) if tt[i,j] > 0 and pd.isna(zones_l[j]) == False])
                closest = np.where(tt[i,:] == closest)[0][0]
                zones_l[i] = zones_l[closest]
        
        # 'assign' nodes to clusters  ################################################################## 
        zones = list(set(zones_l)) # list of unique zone ids    
        clusters = {} # dict that holds all nodes for a zone
        for i in zones:
            clusters[i] = [j for j in range(n) if zones_l[j] == i]
        
        # data per node ###############################################################################

            
        a = [] # lower TW
        b = [] # upper TW
        s = [] # service times
            
        dateFormat = '%Y-%m-%d %H:%M:%S'
        start_time = str(route['date_YYYY_MM_DD']+" "+str(route['departure_time_utc']))
        start_time_obj = datetime.datetime.strptime(start_time, dateFormat)
        start_time = start_time = (start_time_obj.hour * 60 + start_time_obj.minute) * 60 + start_time_obj.second
            
        unlimited_a = 0 # lower TW for unrestricted nodes
        unlimited_b = start_time+2*24*3600 # upper TW for unrestricted nodes: starting time of tour + 2 days
            
        # loop over stops
        for i in nodes:
            s_tmp = 0.0
            if len(package[i])==0: # set parameters for depots
                a.append(start_time)
                b.append(unlimited_b)
                s.append(s_tmp)
                    
            else:   # set parameters for regular stops
                a_stop = [unlimited_a] # in case all TW entries of package are nans
                b_stop = [unlimited_b] 
                    
                # loop over packages of stop
                for p in package[i].values():
                    s_tmp += p['planned_service_time_seconds'] # add service times
                    if isinstance(p['time_window']['start_time_utc'], str):
                        a_package = datetime.datetime.strptime(p['time_window']['start_time_utc'], dateFormat)
                        b_package = datetime.datetime.strptime(p['time_window']['end_time_utc'], dateFormat)
                            
                        # check if time windows stretch til next day 
                        # for TW=[a,_]
                        if int(a_package.day) != int(start_time_obj.day):
                            a_stop.append(((24+a_package.hour) * 60 + a_package.minute) * 60 + a_package.second)
                        else:
                            a_stop.append((a_package.hour * 60 + a_package.minute) * 60 + a_package.second)
                        
                        # for TW=[_,b]
                        if int(b_package.day) != int(start_time_obj.day):
                            b_stop.append(((24+b_package.hour) * 60 + b_package.minute) * 60 + b_package.second)
                        else:
                            b_stop.append((b_package.hour * 60 + b_package.minute) * 60 + b_package.second)
                                
                # set most narrow TW
                s.append(s_tmp)       
                a.append(max(a_stop))
                b.append(min(b_stop))  
        
        # determine centroids of each zone for meta tour
        
        centroids = [] # not actual centroids but node of cluster closest to centroid
        ctrs = np.zeros((len(zones), 2)) # holds coords of the actual centroids
        
        # find cluster node nearest to centroid
        cou = 0 #
        for i in zones:
            if len(clusters[i]) > 1:       
                ctr = [0,0] # centroid of zone i
                ctr[0] = np.mean(points[clusters[i], 0])
                ctr[1] = np.mean(points[clusters[i], 1])
                ctrs[cou, :] = ctr #
                cou+=1 #
                dist = distance.cdist([ctr], points[clusters[i],:])
                min_ind = dist.argmin()
                centroids.append(clusters[i][min_ind])
                
            else:
                centroids.append(clusters[i][0])
                ctrs[cou, :] = points[clusters[i][0],:]#
                cou +=1

        s = np.array(s)
        a = np.array(a)
        b = np.array(b)
        
        # solving routine #############################################################################
        ###############################################################################################
        
        # determine meta tour
        
        #print("starting")
        shortest_distances_clusters={}
        for i in clusters.keys():
            shortest_distances_clusters[i]={}
            for j in clusters.keys():
                if i !=j:
                    best_dist=9999999
                    best_from = None
                    best_to = None
                    for x in clusters[i]:
                        for y in clusters[j]:
                            dist=tt[int(x)][int(y)]
                            if dist<best_dist:
                                best_dist=dist
                                best_from=x
                                best_to=y
                    shortest_distances_clusters[i][j]={"dist":best_dist,"from":best_from,"to":best_to}

        cluster_order=[("Start",clusters["Start"])]
        remaining_cluster=copy.deepcopy(list(clusters.keys()))
        remaining_cluster.remove("Start")

        while len(remaining_cluster)>0:

            best_dist=999999
            best_ind=None
            for ind,r in enumerate(remaining_cluster):
                last=cluster_order[-1][0]
                dist=shortest_distances_clusters[last][r]["dist"]

                if dist <=best_dist:
                    best_dist=dist
                    best_ind=ind

            cluster_order.append((remaining_cluster[best_ind],clusters[remaining_cluster[best_ind]]))
            remaining_cluster.pop(best_ind)
    
        
        #########################################################################
        
        def two_opt_global_clusters(cluster_order, sz_weight, last_zone_weight):
                    improved = True
                    distance = shortest_distances_clusters
                    start_time = time.time()
                    tour=cluster_order
        
                    while improved:
                        if time.time() - start_time >= 10:
                            return tour
        
                        min_i = 9999
                        min_j = 9999
                        change = 0
                        improved = False
                        min_change = 0
                        num_cities = len(tour)
                        # Find the best move
                        for i in range(num_cities - 2):
                            for j in range(i + 2, num_cities - 1):
                                # change = dist(i, j,tour) + dist(i+1, j+1,tour) - dist(i, i+1,tour) - dist(j, j+1,tour)
                                dist_change = distance[tour[i][0]][tour[j][0]]["dist"] + distance[tour[i + 1][0]][tour[j + 1][0]]["dist"] - \
                                            distance[tour[i][0]][
                                                tour[i + 1][0]]["dist"] - distance[tour[j][0]][tour[j + 1][0]]["dist"]
                                
                                ##########################################################                      

                                curr_mz_violation = sum( [ meta_zone(tour[i][0]) != meta_zone(tour[j][0]),
                                                            meta_zone(tour[i+1][0]) != meta_zone(tour[j+1][0]) ] )
                                prev_mz_violation = sum( [ meta_zone(tour[i][0]) != meta_zone(tour[i+1][0]),
                                                            meta_zone(tour[j][0]) != meta_zone(tour[j+1][0]) ] )
                                meta_zone_change = curr_mz_violation - prev_mz_violation
                                

                                if i != 0: 
                                    last_zone_change = sum( [tour[i][-1] != tour[j][-1], tour[i+1][-1] != tour[j+1][-1]] ) \
                                                        -  sum( [tour[i][-1] != tour[i+1][-1], tour[j][-1] != tour[j+1][-1]] )
                                else:
                                    last_zone_change = 0
                                


                                if meta_zone_change != 0:
                                    change = dist_change*(1-sz_weight) + meta_zone_change*100*sz_weight
                                else:
                                    change = dist_change*(1-last_zone_weight) + last_zone_change*100*last_zone_weight

                                
                                ###########################################################
        
                                if change < min_change and change < -0.00000001:
                                    improved = True
                                    min_change = change
                                    min_i, min_j = i, j

                        if min_change < 0:
                            tour[min_i + 1:min_j + 1] = tour[min_i + 1:min_j + 1][::-1]
        
                    return cluster_order
                ##########################################################################

        # apply 2-opts on the clusters 
        cluster_order=two_opt_global_clusters(cluster_order, sz_weight, last_zone_weight)
        curr_dist = sum([shortest_distances_clusters[cluster_order[i][0]][cluster_order[i+1][0]]["dist"] for i in range(len(cluster_order)-1)])
        
        # looping improvement between distance optimization and zone id rules
        new_cluster_order = copy.deepcopy(cluster_order)
        changed= True
        start_time = time.time()
        while (time.time() - start_time <= 20) and changed:
            changed=False
            
            for i in range(1, len(new_cluster_order)-1):
                
                n1 = new_cluster_order[i][0]
                n2 = new_cluster_order[i+1][0]
                rem = [(idx, r[0]) for idx,r in list(enumerate(new_cluster_order))[i+2:]]
                
                # improve following zone id 
                if meta_zone(n1)[0] == meta_zone(n2)[0]:
                    # last zone rule
                    if n1[-1] != n2[-1]:
                        tmp = [idx for idx, r in rem if meta_zone(n1)[0] == meta_zone(r)[0] and n1[-1] == r[-1]]
                        if len(tmp) != 0:
                            new_cluster_order = new_cluster_order[:i+1] \
                                                    + [new_cluster_order[j] for j in tmp] \
                                                        + [new_cluster_order[idx] for idx in range(i+1,len(new_cluster_order)) if idx not in tmp]
                        changed=True
                else:
                    # super zone rule
                    tmp = [idx for idx, r in rem if meta_zone(n1)[0] == meta_zone(r)[0]]   
                    if len(tmp) != 0:
                        new_cluster_order = new_cluster_order[:i+1] \
                                                + [new_cluster_order[j] for j in tmp] \
                                                    + [new_cluster_order[idx] for idx in range(i+1,len(new_cluster_order)) if idx not in tmp]

                        changed=True

            if changed:            
                new_dist = sum([shortest_distances_clusters[new_cluster_order[i][0]][new_cluster_order[i+1][0]]["dist"] for i in range(len(new_cluster_order)-1)])
                # if distance increased < maximum allowed increased => exit
                if new_dist - curr_dist <= max_allowed_dist:
                    cluster_order = copy.deepcopy(new_cluster_order)
                    changed=False
                else:
                    # improve distance
                    new_cluster_order = two_opt_global_clusters(new_cluster_order, sz_weight, last_zone_weight)


        
        meta_tour = []
        for i in cluster_order:
            meta_tour.append(zones.index(i[0]))
        
        # solve cluster paths #########################################################################
        
        # find closest cluster nodes between 2 neighboring clusters in meta tour
        connections = [0] # numbers are not actual node ids but position of node in clusters dictionary 
        for i in range(len(meta_tour)-1):
    
            tt_sel = tt[clusters[ zones[meta_tour[i] ] ], : ] 
            tt_sel = tt_sel[: , clusters[ zones[meta_tour[i+1] ] ] ]
            
            j = np.where( tt_sel == np.min(tt_sel) )
            
            # if j is already used, find second closest        
            if len(clusters[ zones[meta_tour[i] ] ])>1 and \
                clusters[zones[meta_tour[i]]] [connections[-1]] == clusters[zones[meta_tour[i]]] [j[0][0]]:
                tt_sel[j[0][0],:] = 999999
                j = np.where( tt_sel == np.min(tt_sel) )
            
            connections.append(j[0][0])
            connections.append(j[1][0])
    
        # add artificial node as destination of path of the last cluster
        if n not in clusters[ zones[ meta_tour[-1]] ]:
            clusters[ zones[ meta_tour[-1]] ].append(n)
        
        if len(connections)<len(zones)*2:     
            connections.append( len(clusters[ zones[ meta_tour[-1]] ])-1)


        # solve path within clusters ################################################################################################################
        
        # function to determine a score for given tour
        def tour_score(tt, a, b, s, tour, start_t, penalty):#, meta):
            # global penalty
            wait = 0
            delay = 0
            t = start_t + s[tour[0]]
            t_seq = [t]   
            
            for i in range(1, len(tour)):
                t += round(s[tour[i]] + tt[tour[i-1], tour[i]],2)
                t = max(t, a[tour[i]])
                
                wait += max(a[tour[i]]-t, 0)
                delay += max(t-b[tour[i]],0)
            
                t_seq.append(t)
                
            return (1-penalty)*t+penalty*(wait+delay), t_seq, wait, delay
        
        t_final = [start_time] # contains point in time when service is finished at node; for all nodes
        big_tour = [clusters[ zones[ meta_tour[0] ] ][0] ] # contains final tour; KEEP IN MIND: entries are the index of a node in nodes
        
        if len(a) == n:
            a = np.append(a,a[big_tour[0]])
            b = np.append(b,b[big_tour[0]])
            s = np.append(s,s[big_tour[0]])
        
        for i in range(1, len(meta_tour)):
            cn = clusters[ zones[meta_tour[i] ] ] # list of clusters nodes, makes following code more concise
            
            if len(cn) == 1: # 1 node clusters
                big_tour.append(cn[connections[i*2]])
                t_final.append(max(a[big_tour[-1]], t_final[-1] + tt[big_tour[-2], big_tour[-1]] + s[big_tour[-1]]))
                
            elif len(cn) == 2: # 2 node clusters
                big_tour.append(cn[connections[i*2]] )
                t_final.append(max(a[big_tour[-1]], t_final[-1] + tt[big_tour[-2], big_tour[-1]] + s[big_tour[-1]]))
                
                big_tour.append(cn[connections[i*2+1]] )
                t_final.append(max(a[big_tour[-1]], t_final[-1] + tt[big_tour[-2], big_tour[-1]] + s[big_tour[-1]] ))
    
            else: # larger clusters => farthest insertion
                
                # farthest insertion
                sub_tour = [connections[i*2], connections[i*2+1]] # contains path: KEEP IN MIND: entrier are the index of a node in cn
                
                # distance matrix of nodes within cluster
                tt_s = tt[cn, :]
                tt_s = tt_s[:, cn]
                # parameter of nodes within cluster
                a_s = a[cn]
                b_s = b[cn]
                s_s = s[cn]
                
                for i in range(len(cn)):
                    tt_s[i,i] = -1000
        
                tt_sel = copy.deepcopy(tt_s)
                tt_sel[: , sub_tour ]= -1000
                
                test = copy.deepcopy(tt_sel)
                
                rem = list(set([i for i in range(len(cn))])-set(sub_tour))
                for i in range(len(cn)-2):
        
                    # determine farthest centroid 
                    far = np.max(tt_sel[sub_tour,:])
                    far = np.where(tt_sel[sub_tour, :] == far)[1][0]
                    tt_sel[:,far] = -1000
            
                    # insert at best position
                    best = 9999999999
                    b_ind = -1
                    
                    for j in range(1, len(sub_tour)):
                        new_tour = copy.deepcopy(sub_tour)
                        new_tour.insert(j, far)
                        
                        new, t_seq, _, _ = tour_score(tt_s, a_s, b_s, s_s, new_tour, max(a[cn[sub_tour[0]]] ,t_final[-1] + tt[big_tour[-1], cn[sub_tour[0]]]), penalty) 
                
                        if new < best:
                            b_ind = j
                            best_tour = new_tour
                            best = new
                            tseq = t_seq
                    
                    # update rem and sub tour/path
                    rem.remove(far)
                    sub_tour = copy.deepcopy(best_tour)

                t_final += tseq           
                sub_tour2 = [cn[index] for index in sub_tour] # convert to real node ids (index of node in nodes)   
                big_tour += sub_tour2 # add to big tour
            
        big_tour.pop() # pop artificial node
        t_final.pop() # pop artificial node
        
        
        final_score, t_final_val, ff_wait, ff_delay = tour_score(tt, a, b, s, big_tour, start_time, penalty)   
        
        # convert to result format
        out = {}
        for i in range(len(big_tour)):
            out[nodes[big_tour[i]]] = i
        
        return k,out

    except Exception as e:
        print(e)
        print("EXCEPTION CAUGHT!!!!!!!!")
        return k,{}



# Preprocessing ###############################################################################
###############################################################################################

if __name__ == "__main__":

    # Get Directory
    BASE_DIR = path.dirname(path.dirname(path.abspath(__file__))) 


    # Read input data
    print('Reading Input Data')
    # Model Build output
    try:
        model_path=path.join(BASE_DIR, 'data/model_build_outputs/model.json')
        with open(model_path, newline='') as in_file:
            model_build_out = json.load(in_file)
            
    except Exception: # in case build process crashed completely, use fallback parameters 
        print("no model file, use fallback solution instead")
        model_build_out={0: {
                    'sz_weight': {'best': 0.9},
                    'lz_weight': {'best': 0},
                    'penalty': {'best': 0},
                    'max_dist': {'best': 300}},
                1: {
                    'sz_weight': {'best': 0.8},
                    'lz_weight': {'best': 0},
                    'penalty': {'best': 0},
                    'max_dist': {'best': 300}},
                2: {
                    'sz_weight': {'best': 0.8},
                    'lz_weight': {'best': 0},
                    'penalty': {'best': 0},
                    'max_dist': {'best': 300} }
                }

    #check the model output
    for i in range(3): 
        if i not in model_build_out: # check set
            model_build_out.update({i:  {
                    'sz_weight': {'best': 0.9 if i == 0 else 0.8},
                    'lz_weight': {'best': 0},
                    'penalty': {'best': 0},
                    'max_dist': {'best': 300}}
                    })
        else:
            params = ['sz_weight', 'lz_weight', 'penalty', 'max_dist']
            default_vals = [0.9 if i == 1 else 0.8, 0, 0, 300]
            for j in range(len(params)): # check params
                if params[j] not in model_build_out[i]:
                    model_build_out[i].update({params[j]: {'best': default_vals[j]}})
                else:
                    if 'best' not in model_build_out[i][params[j]]:
                        model_build_out[i][params[j]].update({'best': default_vals[j]})

    print(model_build_out)


    # Prediction Routes (Model Apply input)
    prediction_routes_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_route_data.json')
    with open(prediction_routes_path, newline='') as in_file:
        prediction_routes = json.load(in_file)
    # Prediction Travel Times
    prediction_travel_times_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_travel_times.json')
    with open(prediction_travel_times_path, newline='') as in_file:
        prediction_travel_times = json.load(in_file)
    # Prediction Travel Times
    prediction_packages_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_package_data.json')
    with open(prediction_packages_path, newline='') as in_file:
        prediction_packages = json.load(in_file)





    ###################################################################################################

    output = {} # for actual tour output
    count = 1


    # load data ###################################################################################
    # read parameter
    weight_dict = {}
    for key, route in prediction_routes.items():
        weight_dict[key] = {}

        n_sz = len(set([meta_zone(route['stops'][s]['zone_id'])[0] 
                            for s in route['stops'] if str(route['stops'][s]['zone_id']) != 'nan']))
        if n_sz == 1:
            sz_weight = model_build_out[0]['sz_weight']['best']
            lz_weight = model_build_out[0]['lz_weight']['best']
            penalty = model_build_out[0]['penalty']['best']
            max_dist = model_build_out[0]['max_dist']['best']
        elif n_sz == 2:
            sz_weight = model_build_out[1]['sz_weight']['best']
            lz_weight = model_build_out[1]['lz_weight']['best']
            penalty = model_build_out[1]['penalty']['best']
            max_dist = model_build_out[1]['max_dist']['best']
        else:
            sz_weight = model_build_out[2]['sz_weight']['best']
            lz_weight = model_build_out[2]['lz_weight']['best']
            penalty = model_build_out[2]['penalty']['best']
            max_dist = model_build_out[2]['max_dist']['best']
        
        weight_dict[key].update({
                    'sz_weight': sz_weight,
                    'lz_weight': lz_weight,
                    'penalty': penalty,
                    'max_dist': max_dist 
        })
        

    # start solving ###################################################################################
    timer_start = time.time()
    args = [(key, prediction_routes[key], copy.deepcopy(prediction_travel_times[key]), prediction_packages[key],
                timer_start, 
                weight_dict[key]['penalty'], weight_dict[key]['sz_weight'], 
                weight_dict[key]['lz_weight'], weight_dict[key]['max_dist']) for key in list(prediction_routes.keys())]
    pool = mp.Pool(mp.cpu_count())
    result = pool.starmap(solve, args)

    for key,out in result:
        output[key] = {}
        output[key]["proposed"] = {}                    
        output[key]['proposed'] = out

            
    print("Finish solving")

    
    # Write output data
    output_path=path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json')
    with open(output_path, 'w') as out_file:
        json.dump(output, out_file)
        print("Success: The '{}' file has been saved".format(output_path))


    print(f"model apply done after {time.time()-timer_start}")
    with open(path.join(BASE_DIR, 'data/model_apply_outputs/runningtime_apply.json'), 'w') as f:
        f.write(str(time.time()-timer_start))


    print('Done!')  

