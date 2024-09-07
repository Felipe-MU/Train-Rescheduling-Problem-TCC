import pyomo as pyo

def Model3(trains, track, station):
    return

# Model name
model = pyo.AbstractModel('Fixed_route')
# Sets
model.trains = pyo.Set() #train id
model.tracks = pyo.Set() # track id
# model.stations = pyo.Set()
# model.stations_tracks = pyo.Set()
model.routes = pyo.Set(model.trains) # {train: tracks}
model.routes_stations = pyo.Set(model.trains) # {route: [stations tracks]}
model.trains_routes = pyo.Set() # (train, position_track_in_route)
model.trains_stations = pyo.Set() #(train, station)
model.alternative_arcs_id = pyo.Set()  # range of conflicts
model.A_arcs = pyo.Set(model.alternative_arcs_id) # it must contain the arcs and trains [[(train1, train2, track)], ...]
# Parameters
model.M = pyo.Param() # just a big number
model.timeontrack = pyo.Param(model.trains, model.tracks) # {(train, track): timeon}
model.schedule = pyo.Param(model.trains_stations)
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
    track_train1 = list(model.routes[train1])[list(model.routes[train1]).index(track)-1] 
    track_train2 = list(model.routes[train2])[list(model.routes[train2]).index(track)-1]
    return(model.times[train1, track] - model.times[train2, track_train2] >= 0 - model.M * (1 - model.y[arc_id]))

def AlternativeArcs2( model, arc_id):
    train1 = list(model.A_arcs[arc_id])[0]
    train2 = list(model.A_arcs[arc_id])[1]
    track = list(model.A_arcs[arc_id])[2]
    track_train1 = list(model.routes[train1])[list(model.routes[train1]).index(track)-1] 
    track_train2 = list(model.routes[train2])[list(model.routes[train2]).index(track)-1]
    return(model.times[train2, track] - model.times[train1, track_train1] >= 0 - model.M * (model.y[arc_id]))

def DelayConstraint( model, train, station):
    return(model.delays[train, station] >= model.times[train,station] - model.schedule[train,station])


def RouteConstraint(model, train, node):
    N_trackinroute = list(model.routes[train]).index(node)-1
    previous_node = list(model.routes[train])[N_trackinroute]
    if N_trackinroute == 0:
        return (model.times[train, node] >= model.timeontrack[train, previous_node])
    else:
        return (model.times[train, node] - model.times[train, previous_node] >= model.timeontrack[train, previous_node])


model.routeconstraint = pyo.Constraint(model.trains_routes, rule = RouteConstraint)
model.alternativeconstraints1 = pyo.Constraint(model.alternative_arcs_id, rule = AlternativeArcs1)
model.alternativeconstraints2 = pyo.Constraint(model.alternative_arcs_id, rule = AlternativeArcs2)
model.delayconstraint = pyo.Constraint(model.trains_stations, rule = DelayConstraint)

# objective
def objective(model):
    return sum(model.delays[train, station] for train, station in model.trains_stations)

model.obj = pyo.Objective(rule = objective, sense = pyo.minimize)

# solver
cplex = pyo.SolverFactory('cplex_direct')



data = {
    None: {
        'trains' : {None: [train.id for train in trains.values()]},
        'tracks' : {None: [track.id for track in tracks.values()]+[0,94]},
        'routes' : {train.id: train.current_route for train in trains.values()},
        'routes_stations': {train.id: list(train.stations.keys()) for train in trains.values()} ,
        'trains_routes': {None: [(train, node) for train in trains for node in trains[train].current_route[1:]]},
        'trains_stations':{None: [(train, station) for train in trains for station in trains[train].stations]},
        'alternative_arcs_id': {None: list(range(len(problems.alternative_arcs)))},
        'M': {None: 3600},
        'timeontrack' : {(train.id, Node): train.timeontrack[Node] for train in trains.values() for Node in train.timeontrack},
        'schedule' : {(train.id,station): train.begin_schedule[station] for train in trains.values() for station in train.stations},
        'A_arcs' : {arc_id: arc for arc_id, arc in enumerate(problems.alternative_arcs)},
    }}

instance_1 = model.create_instance(data)

result_1 = cplex.solve(instance_1)
status = result_1.solver.termination_condition
print(status)
instance_1.display()


def With_rerouting(self, shortestpath):
        import pyomo.environ as pyo
        from pyomo.util.infeasible import log_infeasible_constraints
        self.preparing_data()
        # finding alternative routes
        self.find_alternative_routes(shortestpath)
        # Model name
        model = pyo.ConcreteModel(f'Rerouting {self.name}')
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
        model.M = pyo.Param(initialize = 3600) # just a big number
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
        
        def ZeroOtherTimes(model, train, routeid, node):
            return model.times[train, routeid, node] <= self.maxtime * model.r[train, routeid]


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

        # fixing past events
        (model.times[train, routeid, 0].fix(0) for train, routeid in model.train_routesid)

        # solver
        # solver = pyo.SolverFactory('cplex_direct')
        solver = pyo.SolverFactory('gurobi', solver_io='python')
        solver.options['TimeLimit'] = 3600
        if shortestpath:
            result_1 = solver.solve(model, logfile = f'Results/Log_File/Rerouting/{self.name}-RR_shortestpaths', tee = True, keepfiles=True)        
        else:
            result_1 = solver.solve(model, logfile = f'Results/Log_File/Rerouting/{self.name}-RR', tee = True, keepfiles=True)
        Mean_number_of_routes = len(self.train_routesid) / len(self.trainsSet)

        if (result_1.solver.status == pyo.SolverStatus.ok) and (result_1.solver.termination_condition == pyo.TerminationCondition.optimal):
            # Do something when the solution in optimal and feasible
            Objective_value = (pyo.value(model.obj))
            return (result_1.solver.wallclock_time, Objective_value, len(self.trainsSet), len(self.alternative_arcs_id), Mean_number_of_routes)
        elif (result_1.solver.termination_condition == pyo.TerminationCondition.infeasible):
            log_infeasible_constraints(model)
            # Do something when model in infeasible
            return (result_1.solver.wallclock_time, 'infeasible', len(self.trainsSet), len(self.alternative_arcs_id), Mean_number_of_routes)
        else:
            # Something else is wrong
            try:
                Objective_value = (pyo.value(model.obj))
                return (result_1.solver.wallclock_time, Objective_value, len(self.trainsSet), len(self.alternative_arcs_id), Mean_number_of_routes)
            except:
                print ('Solver Status: ',  result_1.solver.status)
                return (result_1.solver.wallclock_time, f'{result_1.solver.status}', len(self.trainsSet), len(self.alternative_arcs_id), Mean_number_of_routes)
 
    def Fixed_Warmstart(self):
        import pyomo.environ as pyo
        self.preparing_data()
        # Model name
        model = pyo.ConcreteModel(f'Fixed_route {self.name}')
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
        (model.times[train, 0].fix(0) for train in model.trains)
        
        # Creating Warm start
        for aarc in self.alternative_arcs_id:
            aarc_info = self.A_arcs[aarc] #[[(train1, train2, track)], ...]
            # self.schedule = {(train.id,track): train.begin_schedule[track] for train in self.trains.values() for track in train.current_route[1:-1]}
            if aarc_info[2] < 94:
                if self.trains[aarc_info[0]].begin_schedule[aarc_info[2]] <= self.trains[aarc_info[0]].begin_schedule[aarc_info[2]]:
                    model.y[aarc] = 1
                else:
                    model.y[aarc] = 0