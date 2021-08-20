from os import path
import sys, json, time, numpy as np, pandas as pd, datetime, copy, re, traceback, multiprocessing as mp
from scipy.spatial import distance


        
##########################################################################################################
# code for scoring #######################################################################################
##########################################################################################################

def good_format(file,input_type,filepath):
    '''
    Checks if input dictionary has proper formatting.
    
    Parameters
    ----------
    file : dict
        Dictionary loaded from evaluate input file.
    input_type : str
        Indicates which input of evaluate the current file is. Can be
        "actual," "proposed," "costs," or "invalids."
    filepath : str
        Path from which file was loaded.
    Raises
    ------
    JSONDecodeError
        The file exists and is readable, but it does not have the proper
        formatting for its place in the inputs of evaluate.
    Returns
    -------
    None.
    '''
    
    for route in file:
        if route[:8]!='RouteID_':
            raise JSONDecodeError('Improper route ID in {}. Every route must be denoted by a string that begins with "RouteID_".'.format(filepath))
    if input_type=='proposed' or input_type=='actual':
        for route in file:
            if type(file[route])!=dict or len(file[route])!=1: 
                raise JSONDecodeError('Improper route in {}. Each route ID must map to a dictionary with a single key.'.format(filepath))
            if input_type not in file[route]:
                if input_type=='proposed':
                    raise JSONDecodeError('Improper route in {}. Each route\'s dictionary in a proposed sequence file must have the key, "proposed".'.format(filepath))
                else:
                    raise JSONDecodeError('Improper route in {}. Each route\'s dictionary in an actual sequence file must have the key, "actual".'.format(filepath))
            if type(file[route][input_type])!=dict:
                raise JSONDecodeError('Improper route in {}. Each sequence must be in the form of a dictionary.'.format(filepath))
            num_stops=len(file[route][input_type])
            for stop in file[route][input_type]:
                if type(stop)!=str or len(stop)!=2:
                    raise JSONDecodeError('Improper stop ID in {}. Each stop must be denoted by a two-letter ID string.'.format(filepath))
                stop_num=file[route][input_type][stop]
                if type(stop_num)!=int or stop_num>=num_stops:
                    raise JSONDecodeError('Improper stop number in {}. Each stop\'s position number, x, must be an integer in the range 0<=x<N where N is the number of stops in the route (including the depot).'.format(filepath))
    if input_type=='costs':
        for route in file:
            if type(file[route])!=dict:
                raise JSONDecodeError('Improper matrix in {}. Each cost matrix must be a dictionary.'.format(filepath)) 
            for origin in file[route]:
                if type(origin)!=str or len(origin)!=2:
                    raise JSONDecodeError('Improper stop ID in {}. Each stop must be denoted by a two-letter ID string.'.format(filepath))
                if type(file[route][origin])!=dict:
                    raise JSONDecodeError('Improper matrix in {}. Each origin in a cost matrix must map to a dictionary of destinations'.format(filepath))
                for dest in file[route][origin]:
                    if type(dest)!=str or len(dest)!=2:
                        raise JSONDecodeError('Improper stop ID in {}. Each stop must be denoted by a two-letter ID string.'.format(filepath))
                    if not(type(file[route][origin][dest])==float or type(file[route][origin][dest])==int):
                        raise JSONDecodeError('Improper time in {}. Every travel time must be a float or int.'.format(filepath))
    if input_type=='invalids':
        for route in file:
            if not(type(file[route])==float or type(file[route])==int):
                raise JSONDecodeError('Improper score in {}. Every score in an invalid score file must be a float or int.'.format(filepath))

class JSONDecodeError(Exception):
    pass

def evaluate(actual_routes,submission,cost_matrices, invalid_scores,**kwargs):
    '''
    Calculates score for a submission.
    Parameters
    ----------
    actual_routes_json : str
        filepath of JSON of actual routes.
    submission_json : str
        filepath of JSON of participant-created routes.
    cost_matrices_json : str
        filepath of JSON of estimated times to travel between stops of routes.
    invalid_scores_json : str
        filepath of JSON of scores assigned to routes if they are invalid.
    **kwargs :
        Inputs placed in output. Intended for testing_time_seconds and
        training_time_seconds
    Returns
    -------
    scores : dict
        Dictionary containing submission score, individual route scores, feasibility
        of routes, and kwargs.
    '''

    scores={'submission_score':'x','route_scores':{},'route_feasibility':{}}
    #print("start evaluation")
    count = 0
    for kwarg in kwargs:
        scores[kwarg]=kwargs[kwarg]
    for route in actual_routes:
        #print(str(count))
        count +=1
        if route not in submission:
            scores['route_scores'][route]=invalid_scores[route]
            scores['route_feasibility'][route]=False
        else:
            actual_dict=actual_routes[route]
            actual=route2list(actual_dict)
            try:
                sub_dict=submission[route]
                sub=route2list(sub_dict)
            except:
                scores['route_scores'][route]=invalid_scores[route]
                scores['route_feasibility'][route]=False
            else:
                if isinvalid(actual,sub):
                    scores['route_scores'][route]=invalid_scores[route]
                    scores['route_feasibility'][route]=False
                else:
                    cost_mat=cost_matrices[route]
                    scores['route_scores'][route]=score(actual,sub,cost_mat)
                    scores['route_feasibility'][route]=True
    submission_score=np.mean(list(scores['route_scores'].values()))
    scores['submission_score']=submission_score
    return scores

def score(actual,sub,cost_mat,g=1000):
    '''
    Scores individual routes.
    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    cost_mat : dict
        Cost matrix.
    g : int/float, optional
        ERP gap penalty. Irrelevant if large and len(actual)==len(sub). The
        default is 1000.
    Returns
    -------
    float
        Accuracy score from comparing sub to actual.
    '''
    norm_mat=normalize_matrix(cost_mat)
    return seq_dev(actual,sub)*erp_per_edit(actual,sub,norm_mat,g)

def erp_per_edit(actual,sub,matrix,g=1000):
    '''
    Outputs ERP of comparing sub to actual divided by the number of edits involved
    in the ERP. If there are 0 edits, returns 0 instead.
    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    matrix : dict
        Normalized cost matrix.
    g : int/float, optional
        ERP gap penalty. The default is 1000.
    Returns
    -------
    int/float
        ERP divided by number of ERP edits or 0 if there are 0 edits.
    '''
    total,count=erp_per_edit_helper(actual,sub,matrix,g)
    if count==0:
        return 0
    else:
        return total/count

def erp_per_edit_helper(actual,sub,matrix,g=1000,memo=None):
    '''
    Calculates ERP and counts number of edits in the process.
    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    matrix : dict
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.
    memo : dict, optional
        For memoization. The default is None.
    Returns
    -------
    d : float
        ERP from comparing sub to actual.
    count : int
        Number of edits in ERP.
    '''
    if memo==None:
        memo={}
    actual_tuple=tuple(actual)
    sub_tuple=tuple(sub)
    if (actual_tuple,sub_tuple) in memo:
        d,count=memo[(actual_tuple,sub_tuple)]
        return d,count
    if len(sub)==0:
        d=gap_sum(actual,g)
        count=len(actual)
    elif len(actual)==0:
        d=gap_sum(sub,g)
        count=len(sub)
    else:
        head_actual=actual[0]
        head_sub=sub[0]
        rest_actual=actual[1:]
        rest_sub=sub[1:]
        score1,count1=erp_per_edit_helper(rest_actual,rest_sub,matrix,g,memo)
        score2,count2=erp_per_edit_helper(rest_actual,sub,matrix,g,memo)
        score3,count3=erp_per_edit_helper(actual,rest_sub,matrix,g,memo)
        option_1=score1+dist_erp(head_actual,head_sub,matrix,g)
        option_2=score2+dist_erp(head_actual,'gap',matrix,g)
        option_3=score3+dist_erp(head_sub,'gap',matrix,g)
        d=min(option_1,option_2,option_3)
        if d==option_1:
            if head_actual==head_sub:
                count=count1
            else:
                count=count1+1
        elif d==option_2:
            count=count2+1
        else:
            count=count3+1
    memo[(actual_tuple,sub_tuple)]=(d,count)
    return d,count

def normalize_matrix(mat):
    '''
    Normalizes cost matrix.
    Parameters
    ----------
    mat : dict
        Cost matrix.
    Returns
    -------
    new_mat : dict
        Normalized cost matrix.
    '''
    new_mat=copy.deepcopy(mat)
    time_list=[]
    for origin in mat:
        for destination in mat[origin]:
            time_list.append(mat[origin][destination])
    avg_time=np.mean(time_list)
    std_time=np.std(time_list)
    min_new_time=np.inf
    for origin in mat:
        for destination in mat[origin]:
            old_time=mat[origin][destination]
            new_time=(old_time-avg_time)/std_time
            if new_time<min_new_time:
                min_new_time=new_time
            new_mat[origin][destination]=new_time
    for origin in new_mat:
        for destination in new_mat[origin]:
            new_time=new_mat[origin][destination]
            shifted_time=new_time-min_new_time
            new_mat[origin][destination]=shifted_time
    return new_mat

def gap_sum(path,g):
    '''
    Calculates ERP between two sequences when at least one is empty.
    Parameters
    ----------
    path : list
        Sequence that is being compared to an empty sequence.
    g : int/float
        Gap penalty.
    Returns
    -------
    res : int/float
        ERP between path and an empty sequence.
    '''
    res=0
    for p in path:
        res+=g
    return res

def dist_erp(p_1,p_2,mat,g=1000):
    '''
    Finds cost between two points. Outputs g if either point is a gap.
    Parameters
    ----------
    p_1 : str
        ID of point.
    p_2 : str
        ID of other point.
    mat : dict
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.
    Returns
    -------
    dist : int/float
        Cost of substituting one point for the other.
    '''
    if p_1=='gap' or p_2=='gap':
        dist=g
    else:
        dist=mat[p_1][p_2]
    return dist

def seq_dev(actual,sub):
    '''
    Calculates sequence deviation.
    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    Returns
    -------
    float
        Sequence deviation.
    '''
    actual=actual[1:-1]
    sub=sub[1:-1]
    comp_list=[]
    for i in sub:
        comp_list.append(actual.index(i))
        comp_sum=0
    for ind in range(1,len(comp_list)):
        comp_sum+=abs(comp_list[ind]-comp_list[ind-1])-1
    n=len(actual)
    return (2/(n*(n-1)))*comp_sum

def isinvalid(actual,sub):
    '''
    Checks if submitted route is invalid.
    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    Returns
    -------
    bool
        True if route is invalid. False otherwise.
    '''
    if len(actual)!=len(sub) or set(actual)!=set(sub):
        return True
    elif actual[0]!=sub[0]:
        return True
    else:
        return False

def route2list(route_dict):
    '''
    Translates route from dictionary to list.
    Parameters
    ----------
    route_dict : dict
        Route as a dictionary.
    Returns
    -------
    route_list : list
        Route as a list.
    '''
    if 'proposed' in route_dict:
        stops=route_dict['proposed']
    elif 'actual' in route_dict:
        stops=route_dict['actual']
    route_list=[0]*(len(stops)+1)
    for stop in stops:
        route_list[stops[stop]]=stop
    route_list[-1]=route_list[0]
    return route_list



########################################################################################################
# function to solve instance ###########################################################################
########################################################################################################
def meta_zone(zone):
    m = re.match('[A-Z]+(?=-)', zone)
    higher_meta = m.group(0) if m else zone

    m_ = re.match('[A-Z]-\d+(?=.)', zone)
    meta = m_.group(0) if m_ else zone

    return meta, higher_meta   



# function for solving instance
def solve(k,route, travel_time, package, timer_start, penalty, sz_weight, last_zone_weight, max_allowed_dist):
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





########################################################################################################
# start of model build #################################################################################
########################################################################################################


if __name__ == "__main__":
    # Get Directory
    BASE_DIR = path.dirname(path.dirname(path.abspath(__file__))) 


    # Read input data
    print('Reading Input Data')
    # Build Route Data
    training_routes_path=path.join(BASE_DIR, 'data/model_build_inputs/route_data.json')
    with open(training_routes_path, newline='') as in_file:
        training_routes = json.load(in_file)
    # Build Travel Times
    training_travel_times_path = path.join(BASE_DIR, 'data/model_build_inputs/travel_times.json')
    with open(training_travel_times_path, newline='') as in_file:
        training_travel_times = json.load(in_file)
    # Build Package Data
    training_packages_path = path.join(BASE_DIR, 'data/model_build_inputs/package_data.json')
    with open(training_packages_path, newline='') as in_file:
        training_packages = json.load(in_file)
    # Build Actual Sequences
    training_sequ_path = path.join(BASE_DIR, 'data/model_build_inputs/actual_sequences.json')
    with open(training_sequ_path, newline='') as in_file:
        training_sequences = json.load(in_file)
    # Build invalid Sequences
    training_inv_path = path.join(BASE_DIR, 'data/model_build_inputs/invalid_sequence_scores.json')
    with open(training_inv_path, newline='') as in_file:
        training_invalid = json.load(in_file)



    # write default values in case of error
    output={0: {
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
    model_path=path.join(BASE_DIR, 'data/model_build_outputs/model.json')
    with open(model_path, 'w') as out_file:
        json.dump(output, out_file)
    print("Success: The '{}' file has been saved".format(model_path))



    build_start = time.time()

    try:

        stop_time = 42600
        # sample_size = 2000
        
        # determine keys of high routes 
        high_keys = [key for key,value in training_routes.items() if value["route_score"] == "High"]  
        
        # divide the train set based on the number of super zones
        key_sz = [(k,[meta_zone(training_routes[k]['stops'][s]['zone_id'])[0] for s in training_routes[k]['stops'] if str(training_routes[k]['stops'][s]['zone_id']) != 'nan']) for k in high_keys ]
        keys_sz1 = [k for k,sz_list in key_sz if len(set(sz_list)) == 1]
        keys_sz2 = [k for k,sz_list in key_sz if len(set(sz_list)) == 2]
        keys_sz3 = [k for k,sz_list in key_sz if len(set(sz_list)) >= 3]

        sample_sizes = [int(3000 * len(keys_sz1) / len(key_sz)), 
                        int(3000 * len(keys_sz2) / len(key_sz)),
                        int(3000 * len(keys_sz3) / len(key_sz))]
        # sample_sizes = [10,10,10]
        
        scores = {}
        for i in range(3):
            scores[i] = {}

            ksz = [keys_sz1, keys_sz2, keys_sz3][i]
            
            print('sample size', sample_sizes[i])
            small_set = ksz[:sample_sizes[i]]
            
            tts_small = {selected : training_travel_times[selected] for selected in small_set}
            inv_small = {selected : training_invalid[selected] for selected in small_set}
            act_small = {selected : training_sequences[selected] for selected in small_set}



            ###############################################################################
            print('Test sz weight')
            try:
                # fix other params
                penalty = 0
                lz_weight = 0
                max_dist = 600
                scores[i]['sz_weight'] = {}
                scores[i]['sz_weight']['best'] = 0.9 if i == 0 else 0.8


                sz_vals = [0.5, 0.6, 0.7, 0.8, 0.9]
                
                time_out = False
                count = 1
                total_it = len(sz_vals)
                for sz in sz_vals:
                                            
                    if time.time()-build_start > stop_time:
                        print("Calculation stopped due to time constraint; select best parameter combination")
                        time_out = True
                        break
                    
                
                    print(f"Parameter testing: sz={sz}, lz={lz_weight}, max_dist={max_dist}, TW-penalty={penalty}; {count}/{total_it}")
                    count +=1
                    pro_sequ = {}


                    timer_start = time.time()
                    args = [(key, training_routes[key], copy.deepcopy(training_travel_times[key]), training_packages[key],
                                timer_start, penalty, sz, lz_weight, max_dist) for key in small_set]
                    pool = mp.Pool(mp.cpu_count())
                    result = pool.starmap(solve, args)

                    for key,out in result:
                        pro_sequ[key] = {}
                        pro_sequ[key]["proposed"] = {}                    
                        pro_sequ[key]['proposed'] = out
                        

                    if time_out == False:
                        test_score = evaluate(act_small, pro_sequ, tts_small, inv_small)
                        print('solve ok')
                        print('Set', i, 'sz_weight', sz, 'score', test_score['submission_score'])
                        scores[i]['sz_weight'].update({sz: test_score['submission_score']})

                    print('Set', i, 'sz_weight', sz, 'running time', time.time() - build_start)    

                # select the best sz weight
                params = [val for val in scores[i]['sz_weight'].keys() if val != 'best']
                params.reverse()
                scores[i]['sz_weight']['best'] = min(params, key=lambda x: scores[i]['sz_weight'][x])


            except Exception as e:
                print(e)
                print("Param search fails.")
                scores[i]['sz_weight']['best'] = 0.9 if i == 0 else 0.8


            ###############################################################################
            print('Test lz weight')
            try:
                # fix other params
                penalty = 0
                sz_weight = scores[i]['sz_weight']['best']
                max_dist = 600
                scores[i]['lz_weight'] = {}
                scores[i]['lz_weight']['best'] = 0

                lz_vals = [0, round(sz_weight/2, 1), sz_weight]
                
                time_out = False
                count = 1
                total_it = len(lz_vals)
                for lz in lz_vals:
                                            
                    if time.time()-build_start > stop_time:
                        print("Calculation stopped due to time constraint; select best parameter combination")
                        time_out = True
                        break
                    
                
                    print(f"Parameter testing: sz={sz_weight}, lz={lz}, max_dist={max_dist}, TW-penalty={penalty}; {count}/{total_it}")
                    count +=1
                    pro_sequ = {}

                    timer_start = time.time()
                    args = [(key, training_routes[key], copy.deepcopy(training_travel_times[key]), training_packages[key],
                                timer_start, penalty, sz_weight, lz, max_dist) for key in small_set]
                    pool = mp.Pool(mp.cpu_count())
                    result = pool.starmap(solve, args)

                    for key,out in result:
                        pro_sequ[key] = {}
                        pro_sequ[key]["proposed"] = {}                    
                        pro_sequ[key]['proposed'] = out
                        

                    if time_out == False:
                        test_score = evaluate(act_small, pro_sequ, tts_small, inv_small)
                        print('solve ok')
                        print('Set', i, 'lz_weight', lz, 'score', test_score['submission_score'])
                        scores[i]['lz_weight'].update({lz: test_score['submission_score']})

                    print('Set', i, 'lz_weight', lz, 'running time', time.time() - build_start)    

                # select the best lz weight
                scores[i]['lz_weight']['best'] = min([val for val in scores[i]['lz_weight'].keys() if val != 'best'], key=lambda x: scores[i]['lz_weight'][x])


            except Exception as e:
                print(e)
                print("Param search fails.")
                scores[i]['lz_weight']['best'] = 0



            ###############################################################################
            print('Test penalty weight')
            try:
                # fix other params
                sz_weight = scores[i]['sz_weight']['best']
                lz_weight = scores[i]['lz_weight']['best']
                max_dist = 600
                scores[i]['penalty'] = {}
                scores[i]['penalty']['best'] = 0

                p_vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
                
                time_out = False
                count = 1
                total_it = len(p_vals)
                for p in p_vals:

                    penalty = p
                                            
                    if time.time()-build_start > stop_time:
                        print("Calculation stopped due to time constraint; select best parameter combination")
                        time_out = True
                        break
                    
                
                    print(f"Parameter testing: sz={sz_weight}, lz={lz_weight}, max_dist={max_dist}, TW-penalty={penalty}; {count}/{total_it}")
                    count +=1
                    pro_sequ = {}


                    timer_start = time.time()
                    args = [(key, training_routes[key], copy.deepcopy(training_travel_times[key]), training_packages[key],
                                timer_start, penalty, sz_weight, lz_weight, max_dist) for key in small_set]
                    pool = mp.Pool(mp.cpu_count())
                    result = pool.starmap(solve, args)

                    for key,out in result:
                        pro_sequ[key] = {}
                        pro_sequ[key]["proposed"] = {}                    
                        pro_sequ[key]['proposed'] = out
                        


                    if time_out == False:
                        test_score = evaluate(act_small, pro_sequ, tts_small, inv_small)
                        print('solve ok')
                        print('Set', i, 'penalty', penalty, 'score', test_score['submission_score'])
                        scores[i]['penalty'].update({penalty: test_score['submission_score']})

                    print('Set', i, 'penalty', penalty, 'running time', time.time() - build_start)    

                # select the best penalty weight
                scores[i]['penalty']['best'] = min([val for val in scores[i]['penalty'].keys() if val != 'best'], key=lambda x: scores[i]['penalty'][x])

            except Exception as e:
                print(e)
                print("Param search fails.")
                scores[i]['penalty']['best'] = 0



            ###############################################################################
            print('Test max_dist')
            try:
                # fix other params
                sz_weight = scores[i]['sz_weight']['best']
                lz_weight = scores[i]['lz_weight']['best']
                penalty = scores[i]['penalty']['best']
                scores[i]['max_dist'] = {}
                scores[i]['max_dist']['best'] = 300

                max_dist_vals = [600, 300, 150]
                
                time_out = False
                count = 1
                total_it = len(max_dist_vals)
                for mxd in max_dist_vals:
                                            
                    if time.time()-build_start > stop_time:
                        print("Calculation stopped due to time constraint; select best parameter combination")
                        time_out = True
                        break
                    
                
                    print(f"Parameter testing: sz={sz_weight}, lz={lz_weight}, max_dist={mxd}, TW-penalty={penalty}; {count}/{total_it}")
                    count +=1
                    pro_sequ = {}


                    timer_start = time.time()
                    args = [(key, training_routes[key], copy.deepcopy(training_travel_times[key]), training_packages[key],
                                timer_start, penalty, sz_weight, lz_weight, mxd) for key in small_set]
                    pool = mp.Pool(mp.cpu_count())
                    result = pool.starmap(solve, args)

                    for key,out in result:
                        pro_sequ[key] = {}
                        pro_sequ[key]["proposed"] = {}                    
                        pro_sequ[key]['proposed'] = out
                        


                    if time_out == False:
                        test_score = evaluate(act_small, pro_sequ, tts_small, inv_small)
                        print('solve ok')
                        print('Set', i, 'max_dist', mxd, 'score', test_score['submission_score'])
                        scores[i]['max_dist'].update({mxd: test_score['submission_score']})

                    print('Set', i, 'max_dist', mxd, 'running time', time.time() - build_start)    

                # select the best max_dist
                scores[i]['max_dist']['best'] = min([val for val in scores[i]['max_dist'].keys() if val != 'best'], key=lambda x: scores[i]['max_dist'][x])

            except Exception as e:
                print(e)
                print("Param search fails.")
                scores[i]['max_dist']['best'] = 300


        #######################################################
        output=copy.deepcopy(scores)
        print(output)


    except Exception as e:
        # in case build crashes, use fallback parameters
        print(traceback.format_exc())
        print('error')
        output={0: {
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
        print(output)


    print(f"selected weights: {output}")
    # Write output data
    model_path=path.join(BASE_DIR, 'data/model_build_outputs/model.json')
    with open(model_path, 'w') as out_file:
        json.dump(output, out_file)
        print("Success: The '{}' file has been saved".format(model_path))


    print(f"model build done after {time.time()-build_start}")
    with open(path.join(BASE_DIR, 'data/model_build_outputs/runningtime_build.json'), 'w') as f:
        f.write(str(time.time()-build_start))
