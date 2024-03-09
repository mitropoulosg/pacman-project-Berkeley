from csp import *
import sys
import time

class RLFA(CSP):
    def __init__(self, file):
        file_name=file
        self.constraint_dict = {} #stores constraints
        self.weights={} #stores weights of constraints
        self.prune_history = {} #for bj
        self.order={} #for bj
        # Specify the file path
        file_path1 = "rlfap/var"+ file_name + ".txt"
        self.variables = self.read_variables(file_path1) #stores variables
        self.conflict_set = { var : set() for var in self.variables } #conflict set for every var  

        for var in self.variables:
            self.order[var] = 0

        file_path2 = "rlfap/dom" + file_name +".txt" 
        domains = self.make_domains(file_path1,file_path2) #stores for every var its domain
        file_path = "rlfap/ctr"+ file_name + ".txt"
        neighbors = self.make_neighbors(file_path) #stores vars involved in a constraint (neighbors)
        self.make_constraints(file_path) #this is for function f
        self.make_weight_dict() 
        CSP.__init__(self,self.variables, domains, neighbors, self.f) #call csp and pass the arguments

    def read_variables(self,file_path): #open the file and store variables in a dictionary
        with open(file_path, 'r') as file:
            lines = file.readlines()
            num_variables = int(lines[0].strip())
            variables = list(range(num_variables))
        return variables

#this function opens 2 files and stores to a dictionary for every var its domain
    def make_domains(self,file_path1,file_path2): 
        domains = {}
        domain_dict = {}

        with open(file_path2, 'r') as file:
            lines = file.readlines() 

            for line in lines[1:]:
                values = line.split()
                domain = int(values[0])
                b = int(values[1])
                list_values = list(map(int, values[2:]))
                domain_dict[domain] = list_values #domain dict is a dictionary with key the number of domain and value the domain

        with open(file_path1, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                values = line.split()
                variable = int(values[0])
                domain = int(values[1])
                domains[variable] = domain_dict[domain] #domains dictionary is a dict with key the variable and value the domain associated with
        return domains

#this function opens a file and stores to a dictionary for every variable its neighbor and the opposite
    def make_neighbors(self,file_path):
        neighbors = {}

        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Ξεκινάμε την ανάλυση των γραμμών με τα domains
            for line in lines[1:]:
                values = line.split()
                variable = int(values[0])
                neighbor = int(values[1])

                if variable not in neighbors:
                    neighbors[variable] = [neighbor]
                else:
                    neighbors[variable].append(neighbor)
                if neighbor not in neighbors:
                    neighbors[neighbor] = [variable]
                else:
                    neighbors[neighbor].append(variable)
        return neighbors

#stores to a dictionary the variables involved in a constraint and the actual constraint
    def make_constraints(self,file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                    values = line.split()
                    a = int(values[0])
                    b = int(values[1])
                    c = values[2]
                    d = int(values[3])
                    self.constraint_dict[(a,b)] = (c,d)
                    self.constraint_dict[(b,a)] = (c,d)
                
# f funciton reutrns true or false if the constraint of the given variables is true or false
    def f(self,A,a,B,b):
        (c,d) = self.constraint_dict.get((A,B))
        if c=='=':
             return abs(a-b)==d
        elif c=='>':
            return abs(a-b)>d

#initialize weights dict with 1     
    def make_weight_dict(self):
        for a,b in self.constraint_dict:
            self.weights[(a,b)]=1
        return self.weights


    def restore_conf_sets(self, var):
        """Restore conflict sets of future vars, if var is about to be unassigned."""
        for pruned_var in self.prune_history[var]:
            self.conflict_set[pruned_var] -= {var}

#this function is copy-paste from csp and the changes are describes with comments
def forward_checking(csp, var, value, assignment, removals):
        """Prune neighbor values inconsistent with var=value."""
        csp.support_pruning()
        for B in csp.neighbors[var]:
            if B not in assignment:
                for b in csp.curr_domains[B][:]:
                    if not csp.constraints(var, value, B, b):
                        csp.prune(B, b, removals)

                        if var not in csp.prune_history:
                            csp.prune_history[var]=[B] # Remember prunes caused by var
                        else:
                            csp.prune_history[var].append(B)
                        csp.conflict_set[B].add(var) # var caused a prune in B's domain (adds var to conflict set of B)
                if(not csp.curr_domains[B]): #DOMAIN WIPE OUT 
                    csp.weights[(var,B)]+=1 # ++ weights of the constraint and the opposite
                    csp.weights[(B,var)]+=1
                    csp.conflict_set[var] = csp.conflict_set[var].union(csp.conflict_set[B]) #conflict set of var must append the conflict set of B cause of domain wipe out
                    return False
        return True

#this function is copy-paste from csp and the changes are describes with comments
def AC3(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        revised, checks = revise(csp, Xi, Xj, removals, checks)
        if revised:
            if not csp.curr_domains[Xi]: #DOMAIN WIPE OUT 
                # ++ weights of the constraint and the opposite
                csp.weights[(Xi,Xj)]+=1 
                csp.weights[(Xj,Xi)]+=1
                return False  # CSP is inconsistent
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
    return True  # CSP is satisfiable

#this function is copy-paste from csp and the changes are describes with comments
def mac(csp, var, value, assignment, removals, constraint_propagation=AC3):  #run with AC3
    """Maintain arc consistency."""
    return constraint_propagation(csp, {(X, var) for X in csp.neighbors[var]}, removals)
 

def dom_wdeg(assignment,csp):
        minVal = float('inf')
        bestVar = 0
        for B in csp.variables: #for every var that is not in assignment
            if B not in assignment:
                wdeg=0
                dom=len((csp.curr_domains or csp.domains)[B]) #if cur.domains is empty then dom is len (domains[var])
                for n in csp.neighbors[B]: #for every neighbor of var that is not in assignment
                    if n not in assignment:
                        wdeg=wdeg+csp.weights[(B,n)] #sum of weights
                
                if wdeg==0:
                    ratio=(dom)
                else:
                    ratio=(float)(dom/wdeg)  #dom/wdeg
                if ratio < minVal: #find var with min dom/wdeg
                    minVal = ratio
                    bestVar = B    
        return bestVar

#this function is copy-paste from csp and the changes are describes with comments
def backjumping_search(csp, select_unassigned_variable=first_unassigned_variable,
                        order_domain_values=unordered_domain_values, inference=no_inference, timeout=None):
    visited = set()
    def backjump(assignment, start_time=None):
        if len(assignment) == len(csp.variables):
            return assignment, None
        if start_time is not None and time.time() - start_time > timeout: #for timeout
            return 'Timeout', None
        var = select_unassigned_variable(assignment, csp)
        csp.order[var] +=1 #level of var

        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                csp.prune_history[var] = [] # Remember vars that are in conflict with var
                if inference(csp, var, value, assignment, removals):
                    result, h = backjump(assignment, start_time)
                    if result is not None:
                        return result,None
                    elif var in visited and var != h:
                        csp.conflict_set[var].clear() #clear set
                        visited.discard(var) #not visited anymore
                        csp.restore(removals)
                        csp.unassign(var, assignment)

                        return None, h 

                csp.restore_conf_sets(var) #restore conflict set 
                csp.restore(removals)
        csp.unassign(var, assignment)  #here is a dead end
        visited.add(var)
        h = None
        maxi = 0   
        if len(csp.conflict_set[var]):
            for c in csp.conflict_set[var]:
                if csp.order[c] > maxi: #find h which is the the deppest variable in conflict set of var
                    maxi = csp.order[c]
                    h = c
            csp.conflict_set[h].union(csp.conflict_set[var]) #append to new varibale the conflict set of var 
            csp.conflict_set[h].discard(h) #except h
        return None, h # Current var failed, jump back   

    start_time = None if timeout is None else time.time()
    result,h = backjump({}, start_time)
    if result!=("Timeout"):
        assert result is None or csp.goal_test(result)
    return result

#this function is copy-paste from csp and the changes are describes with comments
def backtracking_search(csp, select_unassigned_variable=first_unassigned_variable,
                        order_domain_values=unordered_domain_values, inference=no_inference, timeout=None):
    def backtrack(assignment, start_time=None):
        if len(assignment) == len(csp.variables):
            return assignment
        if start_time is not None and time.time() - start_time > timeout: #for timeout
            return 'Timeout'
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment, start_time)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        return None

    start_time = None if timeout is None else time.time()
    result = backtrack({}, start_time)
    if(result!='Timeout'):
        assert result is None or csp.goal_test(result)
    return result

#this function is copy-paste from csp and the changes are describes with comments
def min_conflicts(csp, max_steps=100000):
    """Solve a CSP by stochastic Hill Climbing on the number of conflicts."""
    # Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}
    for var in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current
        if i == max_steps-1:      # last time is also the best assignment for algorithm because every time improves the solution (minimizes the conflicts)
            ans = len(conflicted) # get the number of conflicted variables (constraints violated at the moment)
        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    return None



if __name__=='__main__':
    
    file_name = str(sys.argv[1])
    instance=RLFA(file_name)
    algorithm = int(sys.argv[2])

    # Record the start time
    start_time = time.time()
    if(algorithm==1): 
        print("FC-BT")  
        result = backtracking_search(instance,dom_wdeg,unordered_domain_values,forward_checking,500)
    elif(algorithm==2):
        print("MAC-BT")  
        result = backtracking_search(instance,dom_wdeg,unordered_domain_values,mac,500)
    elif(algorithm==3):
        print("FC-CBJ")  
        result = backjumping_search(instance,dom_wdeg,unordered_domain_values,forward_checking,500)
    elif(algorithm==4):
        print("Min_conflicts ")  
        result=min_conflicts(instance)
    else:
        print("No algorithm given")
        sys.exit(1)
    # Record the stop time
    stop_time = time.time()

    # Create a new dictionary with sorted keys
    if(result!=None and result!="Timeout"):
        sorted_dict = {key: result[key] for key in sorted(result)}
    # Print the new dictionary with sorted keys
        print(sorted_dict)
    else:
        print(result)

    print(f"For {len(instance.variables)} variables, assigns = {instance.nassigns}")
    # Calculate and print the elapsed time
    print(f"Elapsed time: {stop_time - start_time} seconds")
    
    

