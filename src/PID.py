# The recipe gives simple implementation of a Discrete Proportional-Integral-Derivative (PID) controller. PID controller gives output value for error between desired reference input and measurement feedback to minimize error value.
# More information: http://en.wikipedia.org/wiki/PID_controller
#
# cnr437@gmail.com
#
#######	Example	#########
#
# p=PID(3.0,0.4,1.2)
# p.setPoint(5.0)
# while True:
#     pid = p.update(measurement_value)
#
#


class PID:
    """
    Discrete PID control    
    """
    def __init__(self, p=2.0, i=0.0, d=1.0, derivator=0, integrator=0, integrator_max=1, integrator_min=-1):

        self.Kp = p
        self.Ki = i
        self.Kd = d
        self.derivator = derivator
        self.integrator = integrator
        self.integrator_max = integrator_max
        self.integrator_min = integrator_min

        self.set_point = 0.0
        self.error = 0.0

        self.p_value = None
        self.i_value = None
        self.d_value = None

    def update(self, current_value):
        """
        Calculate PID output value for given reference input and feedback
        """

        self.error = self.set_point - current_value

        self.p_value = self.Kp * self.error
        self.d_value = self.Kd * (self.error - self.derivator)
        self.derivator = self.error

        self.integrator = self.integrator + self.error

        if self.integrator > self.integrator_max:
            self.integrator = self.integrator_max
        elif self.integrator < self.integrator_min:
            self.integrator = self.integrator_min

        self.i_value = self.integrator * self.Ki

        PID = self.p_value + self.i_value + self.d_value

        return PID
