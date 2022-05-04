class Leg(str):

	DIAG = {
		'fl' : 'hr',
		'fr' : 'hl',
		'hl' : 'fr',
		'hr' : 'fl',
	}
	# counter-clockwise neighbor
	N_CCR = {
		'fl' : 'hl',
		'hl' : 'hr',
		'hr' : 'fr',
		'fr' : 'fl',
	}
	# clockwise neighbor
	N_CR = {
		'fl' : 'fr',
		'fr' : 'hr',
		'hr' : 'hl',
		'hl' : 'fl',
	}
	# front to back
	F2B = {
		'fl' : 'hl',
		'hl' : 'hr',
		'hr' : 'hl',
		'fr' : 'hr',
	}

	def __init__(self, leg_str):
		self.leg_str = leg_str.lower()

	def __str__(self):
		return self.leg_str
	@property
	def neighbor_diag(self):
		return Leg(self.DIAG[self.leg_str])
	@property
	def neighbor_ccr(self):
		return Leg(self.N_CCR[self.leg_str])
	@property
	def neighbor_cr(self):
		return Leg(self.N_CR[self.leg_str])
	@property
	def neighbor_front_to_back(self):
		return Leg(self.F2B[self.leg_str])