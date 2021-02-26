import numpy as np
import mockSQLenv as srv
import const
import sys
import utilities as ut

"""
agent2.py is based on agent
The main difference is that we assume no knowledge of what the responses look like. So a single query will tell us nothing.
"""

class Agent(object):
	def __init__(self, actions, verbose=True):
		self.actions = actions
		self.num_actions = len(actions)

		self.Q = {(): np.ones(self.num_actions)}

		self.verbose = verbose
		self.set_learning_options()
		self.used_actions = []
		self.powerset = None

		self.steps = 0
		self.rewards = 0
		self.total_trials = 0
		self.total_successes = 0


	def set_learning_options(self,exploration=0.2,learningrate=0.1,discount=0.9, max_step = 100):
		self.expl = exploration
		self.lr = learningrate
		self.discount = discount
		self.max_step = max_step

	def _select_action(self, learning = True):
		if (np.random.random() < self.expl and learning):
			return np.random.randint(0,self.num_actions)
		else:
			return np.argmax(self.Q[self.state])


	def step(self):
		self.steps = self.steps + 1

		action = self._select_action()
		self.used_actions.append(action)

		state_resp, reward, termination, debug_msg = self.env.step(action)

		self.rewards = self.rewards + reward
		self._analyze_response(action, state_resp, reward)
		self.terminated = termination
		if(self.verbose): print(debug_msg)

		return


	def _update_state(self, action_nr, response_interpretation):
		"""
		response interpretation is either -1 or 1
		"""
		action_nr += 1
		#The actual response "observed"
		response = response_interpretation*self.response_obscurer

		new_state = list(self.state)

		#This means that this particular response has never been seen
		if(response not in self.seen_responses):
			self.seen_responses.add(response)

			counter = 0
			not_set = True
			#Here we place it in our state
			for act_id in self.smallest_acts:
				if(action_nr < act_id):
					self.output_to_index[response] = counter
					self.smallest_acts = self.smallest_acts[:counter+1] + [action_nr] + self.smallest_acts[counter+1:]
					new_state = new_state[:counter] + [(action_nr,)] + new_state[counter:]
					for key, value in self.output_to_index:
						if(value >= counter):
							self.output_to_index[key] += 1
					not_set = False


					break;
				counter += 1
			if(not_set):
				new_state.append((action_nr,))


		else:
			add_to_index = self.output_to_index[response]
			x = np.sort(list(self.state[add_to_index]) + [action_nr])
			print(x)
			sys.exit()






		#x = list(set(list(self.state) + [*action_nr]))
		#x.sort()
		#x = tuple(x)
		new_state = tuple(new_state)
		print("state2",new_state)
		print("state_old", self.state)

		self.oldstate = self.state
		self.state = new_state

		self.Q[self.state] = self.Q.get(self.state, np.ones(self.num_actions))






	def _analyze_response(self, action, response, reward):
		expl1 = 1 	# SOMETHING
		expl2 = 2 	# NOTHING
		flag  = 3 	#FLAG
		expl3 = 4 	#SOMETHING
		wrong1 = 0 	#NOTHING
		wrong2 = -1 #NOTHING

		#The agent recieves SOMETHING as the response
		if(response==expl1 or response == expl3):
			self._update_state(action, response_interpretation = 1)
			self._update_Q(action, reward)
		#NOTHING2
		elif(response == expl2):
			self._update_state(action, response_interpretation = -1)
			self._update_Q(action, reward)

		elif(response==wrong1 or response == wrong2):
			self._update_state(action, response_interpretation = -1)
			self._update_Q(action, reward)

		elif(response==flag):
			self._update_state(action, response_interpretation = 1)
			self._update_Q(action,reward)
		else:
			print("ILLEGAL RESPONSE")
			sys.exit()

	def _update_Q(self, action, reward):
		best_action_newstate = np.argmax(self.Q[self.state])
		self.Q[self.oldstate][action] = self.Q[self.oldstate][action] + self.lr * (reward + self.discount*self.Q[self.state][best_action_newstate] - self.Q[self.oldstate][action])

	def reset(self,env):
		self.env = env
		self.terminated = False
		self.state = () #empty tuple
		self.oldstate = None
		self.used_actions = []

		self.steps = 0
		self.rewards = 0
		self.response_obscurer = np.random.choice([-1,1])

		#These two is to create the unique state
		self.output_to_index = {}
		self.smallest_acts = []
		self.seen_responses = set([])


	def run_episode(self):
		_,_,self.terminated,s = self.env.reset()
		if(self.verbose): print(s)

		#Limiting the maximimun number of steps we allow the attacker to make to avoid overly long runtimes and extreme action spaces
		while (not(self.terminated)) and self.steps < self.max_step:
			self.step()

		self.total_trials += 1
		if(self.terminated):
			self.total_successes += 1
		return self.terminated

	def run_human_look_episode(self):
		_,_,self.terminated,s = self.env.reset()
		print(s)
		while (not(self.terminated)) and self.steps < self.max_step:
			self.look_step()

		self.total_trials += 1
		if(self.terminated):
			self.total_successes += 1
		return self.terminated

	def look_step(self):
		self.steps = self.steps + 1
		print("step", self.steps)

		print("My state is")
		print(self.state)

		print("My Q row looks like this:")
		print(self.Q[self.state])
		print("Action ranking is")

		print(np.argsort(self.Q[self.state])[::-1])
		action = self._select_action(learning = True)
		print("action equal highest rank",action == np.argsort(self.Q[self.state])[::-1][0])



		state_resp, reward, termination, debug_msg = self.env.step(action)

		self._analyze_response(action, state_resp, reward)
		self.terminated = termination
		print(debug_msg)


if __name__ == "__main__":
	a = Agent(const.actions)
	env = srv.mockSQLenv()
	a.reset(env)
	a.run_episode()
	#a.step()
