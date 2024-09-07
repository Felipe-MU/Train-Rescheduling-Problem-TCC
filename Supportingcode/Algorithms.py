from Supportingcode.Data_reader import Data
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
        # self.Past_times = [(train, node) for train in self.trains for node in self.trains[train].current_route[1:-1] if self.trains[train].begin_schedule[node]<= self.current_time]
        self.timeonstation = {(train.id,station): train.planned_stop[train.stations[station]] for train in self.trains.values() for station in train.stations}


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


