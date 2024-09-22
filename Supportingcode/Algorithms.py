from Supportingcode.Data_reader import Data
from Supportingcode.Classes import Train
class rescheduling_problem:
    def __init__(self, tracks, stations, trains, current_time, maxtime, name) -> None:
        # finding conflicts
        self.trains = trains
        self.locations = {}
        self.conflicts_many_routes = {}
        # creating infra-graph
        self.tracks = tracks
        self.stations = stations
        self.current_time = current_time
        self.maxtime = maxtime
        self.name = name

    def find_Conflicts(self):
        trains = list(self.trains.values())
        for train1 in trains:
            trains.remove(train1)
            for train2 in trains:
                intersections = set(train1.current_route[1:-1]).intersection(train2.current_route[1:-1])
                if len(intersections)>0:
                    self.locations[train1.id,train2.id] = intersections
        self.alternative_arcs = [(train1, train2, track) for train1, train2 in self.locations for track in self.locations[train1, train2]]
        # self.alternative_arcs = [[(node, self.trains[trains[0]].current_route[self.trains[trains[0]].current_route.index(node)-1]), (node, self.trains[trains[1]].current_route[self.trains[trains[1]].current_route.index(node)-1]), trains] for trains in self.locations for node in self.locations[trains]]        
        return self.alternative_arcs
    
    def plot_graph(self, directed):
        import igraph as ig
        n_vertices = len(self.tracks)+2
        self.graph = ig.Graph(n_vertices, ((track.id,next_track) for track in self.tracks.values() for next_track in track.next))
        self.directed_graph = ig.Graph(n_vertices, ((track.id,next_track) for track in self.tracks.values() for next_track in track.next), directed = True)
        Label = list(range(n_vertices))
        for station in self.stations.values():
            for track in station.tracks:
                Label[Label.index(track)] = station.name
        self.graph.vs['label'] = Label
        if directed:
            return ig.plot(self.directed_graph, f"{self.name}.png", bbox=(500, 500), vertex_size = 40 ,vertex_label = self.graph.vs['label'], vertex_color = "rgb(235,144,150)", layout = 'sugiyama')
        else:
            self.graph.simplify()
            return ig.plot(self.graph, f"{self.name}.png", bbox=(2000, 2000), vertex_size = 30 ,vertex_label = self.graph.vs['label'], vertex_color = "rgb(235,144,150)", layout = 'fr')
        
    def find_alternative_routes(self, shortestpath):
        alternative_routes = {train.id: [train.current_route[:2]] for train in self.trains.values()}
        import igraph as ig
        n_vertices = len(self.tracks)+2
        directed_graph = ig.Graph(n_vertices, ((track.id,next_track) for track in self.tracks.values() for next_track in track.next), directed = True)
        Label = list(range(n_vertices))
        for station in self.stations.values():
            for track in station.tracks:
                Label[Label.index(track)] = station.name
        trains_ordered_stations = {train.id: [station for station in train.current_route if station in train.stations] for train in self.trains.values()}
        for train in self.trains.values():
            for index,station in enumerate(trains_ordered_stations[train.id][:len(trains_ordered_stations[train.id])-1]):
                next_station = trains_ordered_stations[train.id][index + 1]
                if shortestpath:
                    alternative_routes[train.id] = [alternative_routes[train.id][x] + directed_graph.get_all_shortest_paths(v = station, to = next_station)[y][1:] for x in range(len(alternative_routes[train.id])) for y in range(len( directed_graph.get_all_shortest_paths(v = station, to = next_station))) if len(set(alternative_routes[train.id][x]).intersection(directed_graph.get_all_shortest_paths(v = station, to = next_station)[y][1:])) == 0]
                else:
                    alternative_routes[train.id] = [alternative_routes[train.id][x] + directed_graph.get_all_simple_paths(v = station, to = next_station)[y][1:] for x in range(len(alternative_routes[train.id])) for y in range(len( directed_graph.get_all_simple_paths(v = station, to = next_station))) if len(set(alternative_routes[train.id][x]).intersection(directed_graph.get_all_simple_paths(v = station, to = next_station)[y][1:])) == 0]
            alternative_routes[train.id] = [alternative_routes[train.id][route] + [94] for route in range(len(alternative_routes[train.id]))]
        self.alternative_routes = {(train, route_id): route for train in alternative_routes for route_id, route in enumerate(alternative_routes[train])} #{(train, routeid): [route]}
        self.alternative_routes_ids = {train: [alternative_route_id for alternative_route_id in range(len(alternative_routes[train]))] for train in alternative_routes} #{train: [routeids]}
        self.train_routesid = [(train, route_id) for train, route_id in self.alternative_routes] #[(train, routeid)]
        self.trains_routes = [(train, route_id, node) for train in alternative_routes for route_id, route in enumerate(alternative_routes[train]) for node in route] #[(train, routeid, node)]
        self.trains_route_after_entrance_node = [(train, route_id, node) for train in alternative_routes for route_id, route in enumerate(alternative_routes[train]) for node in route[1:]] #[(train, routeid, node)]
        # now finding all conflicts
        trains = list(self.trains.keys())
        for train1 in trains:
            trains.remove(train1)
            for train2 in trains:
                for route1id, route1 in enumerate(alternative_routes[train1]):
                    for route2id, route2 in enumerate(alternative_routes[train2]):
                        intersections = set(route1[1:-1]).intersection(route2[1:-1])
                        if len(intersections)>0:
                            self.conflicts_many_routes[train1, route1id, train2, route2id] = intersections
        alternative_arcs = [(train1, route1id, train2, route2id, track) for train1, route1id, train2, route2id in self.conflicts_many_routes for track in self.conflicts_many_routes[train1, route1id, train2, route2id]]
        self.alternative_arcs_id = list(range(len(alternative_arcs)))
        self.A_arcs = {arc_id: arc for arc_id, arc in enumerate(alternative_arcs)}
        # [(train1, routeid_train1, train2, routeid_train2, node)]

    def Current_time_adjust(self):
        for train in self.trains.values():
            for node in train.current_route[1:-1]:
                if train.end_schedule[node]<self.current_time:
                    # train.current_route = train.current_route[train.current_route.index(node)+1:]
                    # print(node, train.id, train.current_route)
                    train.current_route.remove(node)
                    if node in train.stations:
                        train.stations.pop(node)
                # if train.begin_schedule[node]<self.current_time and train.end_schedule[node]>self.current_time:
                #     train.timeontrack[node] = train.timeontrack[node] - (self.current_time - train.begin_schedule[node])
                # print(train.timeontrack)
        Train.define_dummhnode_timeontrack(self.trains)

    def preparing_data(self):
        self.find_Conflicts()
        self.trainsSet = [train.id for train in self.trains.values()]
        self.tracksSet = [track.id for track in self.tracks.values()]+[0,94]
        self.routes = {train.id: train.current_route for train in self.trains.values()}
        self.routes_stations = {train.id: list(train.stations.keys()) for train in self.trains.values()}
        self.trains_routes = [(train.id, node) for train in self.trains.values() for node in train.current_route]
        self.trains_route_after_entrance_node = [(train.id, node) for train in self.trains.values() for node in train.current_route[1:]]
        self.trains_stations = [(train, station) for train in self.trains for station in self.trains[train].stations]
        self.alternative_arcs_id = list(range(len(self.alternative_arcs)))
        self.A_arcs = {arc_id: arc for arc_id, arc in enumerate(self.alternative_arcs)}
        self.timeontrack = {(train.id, Node): train.timeontrack[Node] for train in self.trains.values() for Node in train.timeontrack} 
        self.schedule = {(train.id,track): train.begin_schedule[track] for train in self.trains.values() for track in train.stations}
        # Adding the planned stop to the timeontrack
        self.timeonstation = {(train.id,station): train.planned_stop[train.stations[station]] for train in self.trains.values() for station in train.stations}
        for train in self.trains.values():
            for station in train.stations:
                self.timeontrack[(train.id, station)] +=  train.planned_stop[train.stations[station]]
        # print(self.timeontrack)

    def Model1(self):
        import pyomo.environ as pyo
        self.preparing_data()
        # for train in self.trains.values():
        #     print(train.id, train.begin_schedule)
        # Preprocessing - defining which train goes first
        ordertrain = []
        for id in self.A_arcs:
            if (self.A_arcs[id][1],self.A_arcs[id][0]) in ordertrain:
                self.A_arcs[id] = (self.A_arcs[id][1],self.A_arcs[id][0],self.A_arcs[id][2])
            elif(self.A_arcs[id][0],self.A_arcs[id][1]) in ordertrain:
                self.A_arcs[id] = (self.A_arcs[id][0],self.A_arcs[id][1],self.A_arcs[id][2])
            else:
                if self.trains[self.A_arcs[id][0]].begin_schedule[self.A_arcs[id][2]] > self.trains[self.A_arcs[id][1]].begin_schedule[self.A_arcs[id][2]]:
                    ordertrain.append((self.A_arcs[id][1],self.A_arcs[id][0]))
                    self.A_arcs[id] = (self.A_arcs[id][1],self.A_arcs[id][0],self.A_arcs[id][2])
                elif self.trains[self.A_arcs[id][0]].begin_schedule[self.A_arcs[id][2]] < self.trains[self.A_arcs[id][1]].begin_schedule[self.A_arcs[id][2]]:
                    ordertrain.append((self.A_arcs[id][0],self.A_arcs[id][1]))
                    self.A_arcs[id] = (self.A_arcs[id][0],self.A_arcs[id][1],self.A_arcs[id][2])
            # if self.trains[self.A_arcs[id][0]].begin_schedule[self.A_arcs[id][2]] > self.trains[self.A_arcs[id][1]].begin_schedule[self.A_arcs[id][2]]:
            #     self.A_arcs[id] = (self.A_arcs[id][1],self.A_arcs[id][0],self.A_arcs[id][2])
            # elif self.trains[self.A_arcs[id][0]].begin_schedule[self.A_arcs[id][2]] < self.trains[self.A_arcs[id][1]].begin_schedule[self.A_arcs[id][2]]:
            #     self.A_arcs[id] = (self.A_arcs[id][0],self.A_arcs[id][1],self.A_arcs[id][2])
        # With this pre-processing, the train that comes first is in front of the other in the alternative arc
        # Model name
        model = pyo.ConcreteModel(f'Model1 {self.name}')
        # Sets
        model.trains = pyo.Set(initialize = self.trainsSet) #train id
        model.tracks = pyo.Set(initialize = self.tracksSet) # track id
        model.routes = pyo.Set(model.trains, initialize = self.routes) # {train: tracks}
        # model.routes_stations = pyo.Set(model.trains, initialize = self.routes_stations) # {route: [stations tracks]}
        model.trains_routes = pyo.Set(initialize = self.trains_routes) # (train, position_track_in_route)
        model.trains_route_after_entrance_node = pyo.Set(initialize = self.trains_route_after_entrance_node)
        model.trains_stations = pyo.Set(initialize = self.trains_stations) #(train, station)
        model.alternative_arcs_id = pyo.Set(initialize = self.alternative_arcs_id)  # range of conflicts
        # Parameters
        # model.M = pyo.Param(initialize = self.maxtime) # just a big number
        model.A_arcs = pyo.Param(model.alternative_arcs_id, within = pyo.Any, initialize = self.A_arcs) # it must contain the arcs and trains [[(train1, train2, track)], ...]
        model.timeontrack = pyo.Param(model.trains, model.tracks, initialize = self.timeontrack) # {(train, track): timeon}
        model.schedule = pyo.Param(model.trains_stations, initialize = self.schedule)
        # Variables
        model.times = pyo.Var(model.trains_routes, domain = pyo.NonNegativeReals) 
        model.delays = pyo.Var(model.trains_stations, domain = pyo.NonNegativeReals)
        # Alternative arcs variable
        # model.y = pyo.Var(model.alternative_arcs_id, domain = pyo.Binary)
        # constraints
        # def AlternativeArcs1( model, arc_id):
        #     train1 = list(model.A_arcs[arc_id])[0]
        #     train2 = list(model.A_arcs[arc_id])[1]
        #     track = list(model.A_arcs[arc_id])[2]
        #     # track_train1 = list(model.routes[train1])[list(model.routes[train1]).index(track)-1] 
        #     track_train2 = list(model.routes[train2])[list(model.routes[train2]).index(track)+1]
        #     return(model.times[train1, track] - model.times[train2, track_train2] >= 0 - model.M * (1 - model.y[arc_id]))

        def AlternativeArcs2( model, arc_id):
            train1 = list(model.A_arcs[arc_id])[0]
            train2 = list(model.A_arcs[arc_id])[1]
            track = list(model.A_arcs[arc_id])[2]
            track_train1 = list(model.routes[train1])[list(model.routes[train1]).index(track)+1] 
            # track_train2 = list(model.routes[train2])[list(model.routes[train2]).index(track)-1]
            return(model.times[train2, track] - model.times[train1, track_train1] >= 0)

        def DelayConstraint( model, train, station):
            return(model.delays[train, station] >= model.times[train,station] - model.schedule[train,station])


        def RouteConstraint(model, train, node):
            N_trackinroute = list(model.routes[train]).index(node)-1
            previous_node = list(model.routes[train])[N_trackinroute]
            return (model.times[train, node] - model.times[train, previous_node] >= model.timeontrack[train, previous_node])


        model.routeconstraint = pyo.Constraint(model.trains_route_after_entrance_node, rule = RouteConstraint)
        # model.alternativeconstraints1 = pyo.Constraint(model.alternative_arcs_id, rule = AlternativeArcs1)
        model.alternativeconstraints2 = pyo.Constraint(model.alternative_arcs_id, rule = AlternativeArcs2)
        model.delayconstraint = pyo.Constraint(model.trains_stations, rule = DelayConstraint)

        # objective
        def objective(model):
            return sum(model.delays[train, station] for train, station in model.trains_stations)
            # return sum(model.delays[train, station] + model.time_lost_on_station[train,station] for train, station in model.trains_stations)

        model.obj = pyo.Objective(rule = objective, sense = pyo.minimize)

        # fixing past events
        # (model.times[train, 0].fix(0) for train in model.trains)
        # (model.times[train, node].fix(model.schedule[train, node]) for train, node in model.Past_times)
        # from pyomo.contrib.parmest.utils import ipopt_solve_with_stats

        # solver
        # solver = pyo.SolverFactory('cplex_direct')
        solver = pyo.SolverFactory('gurobi', solver_io='python')
        # solver.solve(model)
        # model.display()
        result_1 = solver.solve(model, tee = True)
        # Objective_value = (pyo.value(model.obj))
        return (model)

        # if Warmstart:
        #     result_1 = solver.solve(model)
        #     return model
        # else:
        #     result_1 = solver.solve(model, logfile = f'Results/Log_File/Fixed Routes/{self.name}-F', tee = True, keepfiles=True)
        #     Objective_value = (pyo.value(model.obj))
        #     return (result_1.solver.wallclock_time, Objective_value, len(self.trainsSet), len(self.alternative_arcs_id), 1)
        

    def Model2(self):
        import pyomo.environ as pyo
        self.preparing_data()
        # Model name
        model = pyo.ConcreteModel(f'Model2 {self.name}')
        # Sets
        model.trains = pyo.Set(initialize = self.trainsSet) #train id
        model.tracks = pyo.Set(initialize = self.tracksSet) # track id
        model.routes = pyo.Set(model.trains, initialize = self.routes) # {train: tracks}
        # model.routes_stations = pyo.Set(model.trains, initialize = self.routes_stations) # {route: [stations tracks]}
        model.trains_routes = pyo.Set(initialize = self.trains_routes) # (train, position_track_in_route)
        model.trains_route_after_entrance_node = pyo.Set(initialize = self.trains_route_after_entrance_node)
        model.trains_stations = pyo.Set(initialize = self.trains_stations) #(train, station)
        model.alternative_arcs_id = pyo.Set(initialize = self.alternative_arcs_id)  # range of conflicts
        # Parameters
        model.M = pyo.Param(initialize = self.maxtime) # just a big number
        model.A_arcs = pyo.Param(model.alternative_arcs_id, within = pyo.Any, initialize = self.A_arcs) # it must contain the arcs and trains [[(train1, train2, track)], ...]
        model.timeontrack = pyo.Param(model.trains, model.tracks, initialize = self.timeontrack) # {(train, track): timeon}
        model.schedule = pyo.Param(model.trains_stations, initialize = self.schedule)
        # Variables
        model.times = pyo.Var(model.trains_routes, domain = pyo.NonNegativeReals) 
        model.delays = pyo.Var(model.trains_stations, domain = pyo.NonNegativeReals)
        # Alternative arcs variable
        model.y = pyo.Var(model.alternative_arcs_id, domain = pyo.Binary)
        # constraints
        def AlternativeArcs1( model, arc_id):
            train1 = list(model.A_arcs[arc_id])[0]
            train2 = list(model.A_arcs[arc_id])[1]
            track = list(model.A_arcs[arc_id])[2]
            # track_train1 = list(model.routes[train1])[list(model.routes[train1]).index(track)-1] 
            track_train2 = list(model.routes[train2])[list(model.routes[train2]).index(track)+1]
            return(model.times[train1, track] - model.times[train2, track_train2] >= 0 - model.M * (1 - model.y[arc_id]))

        def AlternativeArcs2( model, arc_id):
            train1 = list(model.A_arcs[arc_id])[0]
            train2 = list(model.A_arcs[arc_id])[1]
            track = list(model.A_arcs[arc_id])[2]
            track_train1 = list(model.routes[train1])[list(model.routes[train1]).index(track)+1] 
            # track_train2 = list(model.routes[train2])[list(model.routes[train2]).index(track)-1]
            return(model.times[train2, track] - model.times[train1, track_train1] >= 0 - model.M * (model.y[arc_id]))

        def DelayConstraint( model, train, station):
            return(model.delays[train, station] >= model.times[train,station] - model.schedule[train,station])


        def RouteConstraint(model, train, node):
            N_trackinroute = list(model.routes[train]).index(node)-1
            previous_node = list(model.routes[train])[N_trackinroute]
            return (model.times[train, node] - model.times[train, previous_node] >= model.timeontrack[train, previous_node])


        model.routeconstraint = pyo.Constraint(model.trains_route_after_entrance_node, rule = RouteConstraint)
        model.alternativeconstraints1 = pyo.Constraint(model.alternative_arcs_id, rule = AlternativeArcs1)
        model.alternativeconstraints2 = pyo.Constraint(model.alternative_arcs_id, rule = AlternativeArcs2)
        model.delayconstraint = pyo.Constraint(model.trains_stations, rule = DelayConstraint)

        # objective
        def objective(model):
            return sum(model.delays[train, station] for train, station in model.trains_stations)
            # return sum(model.delays[train, station] + model.time_lost_on_station[train,station] for train, station in model.trains_stations)

        model.obj = pyo.Objective(rule = objective, sense = pyo.minimize)

        # fixing past events
        # (model.times[train, 0].fix(0) for train in model.trains)
        # (model.times[train, node].fix(model.schedule[train, node]) for train, node in model.Past_times)
        # from pyomo.contrib.parmest.utils import ipopt_solve_with_stats

        # solver
        # solver = pyo.SolverFactory('cplex_direct')
        solver = pyo.SolverFactory('gurobi', solver_io='python')
        # solver.solve(model)
        # model.display()
        result_1 = solver.solve(model, tee = True)
        Objective_value = (pyo.value(model.obj))
        return (model)

        # if Warmstart:
        #     result_1 = solver.solve(model)
        #     return model
        # else:
        #     result_1 = solver.solve(model, logfile = f'Results/Log_File/Fixed Routes/{self.name}-F', tee = True, keepfiles=True)
        #     Objective_value = (pyo.value(model.obj))
        #     return (result_1.solver.wallclock_time, Objective_value, len(self.trainsSet), len(self.alternative_arcs_id), 1)
        
    def Model3(self):
        import pyomo.environ as pyo
        from pyomo.util.infeasible import log_infeasible_constraints
        self.preparing_data()
        # finding alternative routes
        self.find_alternative_routes(True)
        # Model name
        model = pyo.ConcreteModel(f'Model3 {self.name}')
        # Sets
        model.trains = pyo.Set(initialize = self.trainsSet) #train id
        model.tracks = pyo.Set(initialize = self.tracksSet) # track id
        model.train_routesid = pyo.Set(initialize = self.train_routesid) #[(train, routeid)]
        model.trains_routes = pyo.Set(initialize = self.trains_routes) # (train, routeid, node)
        model.trains_route_after_entrance_node = pyo.Set(initialize = self.trains_route_after_entrance_node) #(train, routeid, node)
        model.trains_stations = pyo.Set(initialize = self.trains_stations) #(train, station)
        model.alternative_arcs_id = pyo.Set(initialize = self.alternative_arcs_id)  # range of conflicts [routeids]
        model.routes = pyo.Set(model.train_routesid, initialize = self.alternative_routes) # {(train, routeid): [tracks]}
        model.routes_ids = pyo.Set(model.trains, initialize = self.alternative_routes_ids) #{train: [routeids]}       
        # Parameters 
        model.M = pyo.Param(initialize = self.maxtime) # just a big number
        model.A_arcs = pyo.Param(model.alternative_arcs_id, within = pyo.Any, initialize = self.A_arcs) # it must contain the arcs and trains and routeids 
        # [(train1, routeid_train1, train2, routeid_train2, track)]
        model.timeontrack = pyo.Param(model.trains, model.tracks, initialize = self.timeontrack) # {(train, track): timeon}
        model.schedule = pyo.Param(model.trains_stations, initialize = self.schedule)
        # Variables
        model.times = pyo.Var(model.trains_routes, domain = pyo.NonNegativeReals) 
        model.delays = pyo.Var(model.trains_stations, domain = pyo.NonNegativeReals)
        # Alternative arcs variable
        model.y = pyo.Var(model.alternative_arcs_id, domain = pyo.Binary)
        # Alternative routes variable
        model.r = pyo.Var(model.train_routesid, domain = pyo.Binary)
        # constraints
        def AlternativeArcs1( model, arc_id):
            train1 = list(model.A_arcs[arc_id])[0]
            routeid1 = list(model.A_arcs[arc_id])[1]
            train2 = list(model.A_arcs[arc_id])[2]
            routeid2 = list(model.A_arcs[arc_id])[3]
            track = list(model.A_arcs[arc_id])[4]
            # track_train1 = list(model.routes[train1, routeid1])[list(model.routes[train1, routeid1]).index(track)-1] 
            track_train2 = list(model.routes[train2, routeid2])[list(model.routes[train2, routeid2]).index(track)+1]
            return(model.times[train1, routeid1, track] - model.times[train2, routeid2, track_train2] + model.M * (1 - model.r[train1, routeid1]) + model.M * (1 - model.r[train2, routeid2]) + model.M * (1 - model.y[arc_id])) >= 0 

        def AlternativeArcs2( model, arc_id):
            train1 = list(model.A_arcs[arc_id])[0]
            routeid1 = list(model.A_arcs[arc_id])[1]
            train2 = list(model.A_arcs[arc_id])[2]
            routeid2 = list(model.A_arcs[arc_id])[3]
            track = list(model.A_arcs[arc_id])[4]
            track_train1 = list(model.routes[train1, routeid1])[list(model.routes[train1, routeid1]).index(track)+1] 
            # track_train2 = list(model.routes[train2, routeid2])[list(model.routes[train2, routeid2]).index(track)-1]
            return( model.times[train2, routeid2, track] - model.times[train1, routeid1, track_train1] + model.M * (1 - model.r[train1, routeid1]) + model.M * (1 - model.r[train2, routeid2]) + model.M * (model.y[arc_id])) >= 0 

        def DelayConstraint( model, train, station):
            return(model.delays[train, station] >= sum(model.times[train, routeid, station] for routeid in model.routes_ids[train]) - model.schedule[train,station])


        def RouteConstraint(model, train, routeid, node):
            N_trackinroute = list(model.routes[train, routeid]).index(node)-1
            previous_node = list(model.routes[train, routeid])[N_trackinroute]
            return (model.times[train, routeid, node] - model.times[train, routeid, previous_node] + model.M * (1 - model.r[train, routeid]) >= model.timeontrack[train, previous_node])
        
        def Route_constraint(model, train):
            return sum(model.r[train, routeid] for routeid in model.routes_ids[train]) == 1
        
        # def ZeroOtherTimes(model, train, routeid, node):
        #     return model.times[train, routeid, node] <= self.maxtime * model.r[train, routeid]


        model.routeconstraint = pyo.Constraint(model.trains_route_after_entrance_node, rule = RouteConstraint)
        model.alternativeconstraints1 = pyo.Constraint(model.alternative_arcs_id, rule = AlternativeArcs1)
        model.alternativeconstraints2 = pyo.Constraint(model.alternative_arcs_id, rule = AlternativeArcs2)
        model.delayconstraint = pyo.Constraint(model.trains_stations, rule = DelayConstraint)
        model.routeselectionconstraint = pyo.Constraint(model.trains, rule = Route_constraint)
        # model.zerootherconstraints = pyo.Constraint(model.trains_route_after_entrance_node, rule = ZeroOtherTimes)

        # objective
        def objective(model):
            return sum(model.delays[train, station] for train, station in model.trains_stations)
            # return sum(model.delays[train, station] + model.time_lost_on_station[train,station] for train, station in model.trains_stations)

        model.obj = pyo.Objective(rule = objective, sense = pyo.minimize)

        # solver
        # solver = pyo.SolverFactory('cplex_direct')
        solver = pyo.SolverFactory('gurobi', solver_io='python')
        solver.options['TimeLimit'] = 90
        result_1 = solver.solve(model, tee = True)
        return model        
        
        # if (result_1.solver.status == pyo.SolverStatus.ok) and (result_1.solver.termination_condition == pyo.TerminationCondition.optimal):
        #     # Do something when the solution in optimal and feasible
        #     Objective_value = (pyo.value(model.obj))
        #     return (result_1.solver.wallclock_time, Objective_value, len(self.trainsSet), len(self.alternative_arcs_id), Mean_number_of_routes)
        # elif (result_1.solver.termination_condition == pyo.TerminationCondition.infeasible):
        #     log_infeasible_constraints(model)
        #     # Do something when model in infeasible
        #     return (result_1.solver.wallclock_time, 'infeasible', len(self.trainsSet), len(self.alternative_arcs_id), Mean_number_of_routes)
        # else:
        #     # Something else is wrong
        #     try:
        #         Objective_value = (pyo.value(model.obj))
        #         return (result_1.solver.wallclock_time, Objective_value, len(self.trainsSet), len(self.alternative_arcs_id), Mean_number_of_routes)
        #     except:
        #         print ('Solver Status: ',  result_1.solver.status)
        #         return (result_1.solver.wallclock_time, f'{result_1.solver.status}', len(self.trainsSet), len(self.alternative_arcs_id), Mean_number_of_routes)
 

class experiment:
    def __init__(self, model, data_files, local, data_name) -> None:
        for data_file in data_files:
            self.instance_name = data_file.split('.')[0]
            experiment_data = Data(data_file, local)
            tracks, stations, trains, current_time, maxtime = experiment_data.read_from_CSV()
            problem =  rescheduling_problem(tracks, stations, trains, current_time, maxtime, self.instance_name)
            if(model == 'F'):
                time, objective, numberTrains, numberAlternativeArcs, Mean_number_of_routes = problem.fixed_route(False)
                self.save_results( model, data_name, time, objective, numberTrains, numberAlternativeArcs, Mean_number_of_routes)
            elif (model == 'FWS'):
                time, objective, numberTrains, numberAlternativeArcs, Mean_number_of_routes = problem.Fixed_Warmstart()
                self.save_results( model, data_name, time, objective, numberTrains, numberAlternativeArcs, Mean_number_of_routes)
            elif(model == 'RR'):
                time, objective, numberTrains, numberAlternativeArcs, Mean_number_of_routes = problem.With_rerouting(False)
                self.save_results( model, data_name, time, objective, numberTrains, numberAlternativeArcs, Mean_number_of_routes)
            elif(model == 'RRshortest'):
                time, objective, numberTrains, numberAlternativeArcs, Mean_number_of_routes = problem.With_rerouting(True)
                self.save_results( model, data_name, time, objective, numberTrains, numberAlternativeArcs, Mean_number_of_routes)
            elif(model == 'RRWS'):
                time, objective, numberTrains, numberAlternativeArcs, Mean_number_of_routes = problem.With_rerouting_and_warmstart(False)
                self.save_results( model, data_name, time, objective, numberTrains, numberAlternativeArcs, Mean_number_of_routes)
            elif(model == 'Bi'):
                time, objective, numberTrains, numberAlternativeArcs, Mean_number_of_routes = problem.Bi_objective()
                self.save_results( model, data_name, time, objective, numberTrains, numberAlternativeArcs, Mean_number_of_routes)
            else:
                print('Model not implemented')
        
    def save_results(self, model, Data_name, time, objective, numberTrains, numberAlternativeArcs, mean_number_of_routes):
        import pandas as pd
        import os
        results = {'Number of trains': numberTrains, 'Number of alternative arcs': numberAlternativeArcs, 'Model Time': time, 'Objective Value': objective, 'Mean number of routes': mean_number_of_routes}
        if not os.path.exists('Results/ExperimentsResults.xlsx'):
            df = pd.DataFrame(results, index= [self.instance_name])
            df.to_excel('Results/ExperimentsResults.xlsx', sheet_name=f'{Data_name}-{model}')
        else:
            with pd.ExcelWriter('Results/ExperimentsResults.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer: 
                # timeseries.to_excel(writer, timeseriesSheetName)
                if not f'{Data_name}-{model}' in pd.ExcelFile('Results/ExperimentsResults.xlsx').sheet_names:
                    df = pd.DataFrame(results, index= [self.instance_name])
                    df.to_excel(writer, sheet_name=f'{Data_name}-{model}')
                else:
                    atual_df = pd.read_excel('Results/ExperimentsResults.xlsx', sheet_name=f'{Data_name}-{model}', index_col=0)
                    # new_row = pd.DataFrame(results, index= [self.instance_name])
                    atual_df.loc[self.instance_name] = results
                    atual_df.to_excel(writer, sheet_name=f'{Data_name}-{model}')


