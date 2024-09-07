import pyomo as pyo

def Model2(trains, track, station):
    return

# Model name
model = pyo.ConcreteModel('Fixed_route')
# Sets
model.trains = pyo.Set(initialize = [train.id for train in trains.values()]) #train id
model.tracks = pyo.Set(initialize = [track.id for track in tracks.values()]+[0,94]) # track id
# model.stations = pyo.Set()
# model.stations_tracks = pyo.Set()
model.routes = pyo.Set(model.trains, initialize = {train.id: train.current_route for train in trains.values()}) # {train: tracks}
model.routes_stations = pyo.Set(model.trains, initialize = {train.id: list(train.stations.keys()) for train in trains.values()}) # {route: [stations tracks]}
model.trains_routes = pyo.Set(initialize = [(train, node) for train in trains for node in trains[train].current_route[1:]]) # (train, position_track_in_route)
model.trains_stations = pyo.Set(initialize = [(train, station) for train in trains for station in trains[train].stations]) #(train, station)
model.alternative_arcs_id = pyo.Set(initialize = list(range(len(problems.alternative_arcs))))  # range of conflicts
model.A_arcs = pyo.Set(model.alternative_arcs_id, initialize = {arc_id: arc for arc_id, arc in enumerate(problems.alternative_arcs)}) # it must contain the arcs and trains [[(train1, train2, track)], ...]
# Parameters
model.M = pyo.Param(initialize = 3600) # just a big number
model.timeontrack = pyo.Param(model.trains, model.tracks, initialize = {(train.id, Node): train.timeontrack[Node] for train in trains.values() for Node in train.timeontrack}) # {(train, track): timeon}
model.schedule = pyo.Param(model.trains_stations, initialize = {(train.id,station): train.begin_schedule[station] for train in trains.values() for station in train.stations})
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



result_1 = cplex.solve(model)
status = result_1.solver.termination_condition
print(status)
model.display()



    def fixed_route(self, Warmstart):
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
        # (model.times[train, node].fix(model.schedule[train, node]) for train, node in model.Past_times)
        # from pyomo.contrib.parmest.utils import ipopt_solve_with_stats

        # solver
        # solver = pyo.SolverFactory('cplex_direct')
        solver = pyo.SolverFactory('gurobi', solver_io='python')

        # model.display()
        if Warmstart:
            result_1 = solver.solve(model)
            return model
        else:
            result_1 = solver.solve(model, logfile = f'Results/Log_File/Fixed Routes/{self.name}-F', tee = True, keepfiles=True)
            Objective_value = (pyo.value(model.obj))
            return (result_1.solver.wallclock_time, Objective_value, len(self.trainsSet), len(self.alternative_arcs_id), 1)