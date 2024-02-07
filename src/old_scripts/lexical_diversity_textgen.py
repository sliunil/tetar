
class TextGenerator:
    """A class to create artificial texts, following a zipf-mandelbrot 
    distribution parametrized by the slope only. The intercept and the shift 
    are deduced from data"""
    
    # Constructor
    def __init__(self):
        self.slopes = np.empty((0))
        self.intercepts = np.empty((0))
        self.shifts = np.empty((0))
        self.model_sl_in = LinearRegression()
        self.model_sl_sh = LinearRegression()
        self.model_in_sh = LinearRegression()
        
    # Fit the models
    def fit(self, slopes, intercepts, shifts):
        self.slopes = slopes
        self.intercepts = intercepts
        self.shifts = shifts
        self.model_sl_in.fit(slopes.reshape(-1, 1), intercepts)
        if np.sum(self.shifts) > 1e-10:
            self.model_sl_sh.fit(np.log(-slopes).reshape(-1, 1), 
                                 np.log(np.abs(shifts)))
            self.model_in_sh.fit(np.log(intercepts).reshape(-1, 1), 
                                 np.log(np.abs(shifts)))
            

    # Plot the relationships
    def plot(self, groups=None, which_cmap="binary"):
        
        # Group colors 
        if groups is not None:
            gr_fact = np.unique(groups, return_inverse=True)
            markers = ['o', 'x', 'v']
            fillstyles = ['full', 'none', 'none']
            cmap = cm.get_cmap(which_cmap, len(gr_fact[0]) + 1)
            
        # Intercepts from slopes
        sorted_sl = np.sort(self.slopes)
        in_from_sl = self.model_sl_in.predict(sorted_sl.reshape(-1, 1))
        in_sl_fig, in_sl_ax = plt.subplots()
        if groups is not None:
            for id_gr, gr in enumerate(gr_fact[0]): 
                in_sl_ax.scatter(self.slopes[gr_fact[1] == id_gr], 
                                 self.intercepts[gr_fact[1] == id_gr], 
                                 c=cmap(id_gr+1), 
                                 marker=markers[id_gr],
                                 facecolors=fillstyles[id_gr], 
                                 label=gr)
            in_sl_ax.legend()
        else:
            in_sl_ax.scatter(self.slopes, self.intercepts)
        in_sl_ax.plot(sorted_sl, in_from_sl, color="black")
        in_sl_ax.set_xlabel("Slope")
        in_sl_ax.set_ylabel("Intercept")
        
        # Shifts from slopes
        sh_sl_fig, sh_sl_ax = plt.subplots()
        if np.sum(self.shifts) > 1e-10:
            sh_from_sl = np.exp(
                self.model_sl_sh.predict(np.log(-sorted_sl).reshape(-1, 1)))
            sh_sl_ax.plot(sorted_sl, sh_from_sl, color="black")
        if groups is not None:
            for id_gr, gr in enumerate(gr_fact[0]): 
                sh_sl_ax.scatter(self.slopes[gr_fact[1] == id_gr], 
                                self.shifts[gr_fact[1] == id_gr], 
                                c=cmap(id_gr+1), 
                                marker=markers[id_gr],
                                facecolors=fillstyles[id_gr], 
                                label=gr)
            sh_sl_ax.legend()
        else:
            sh_sl_ax.scatter(self.slopes, self.shifts)
        sh_sl_ax.set_xlabel("Slope")
        sh_sl_ax.set_ylabel("Shift")
            

        # Shifts from intercepts
        sh_in_fig, sh_in_ax = plt.subplots()
        sorted_in = np.sort(self.intercepts)
        if np.sum(self.shifts) > 1e-10:
            sh_from_in = np.exp(
                self.model_in_sh.predict(np.log(sorted_in).reshape(-1, 1)))
            sh_in_ax.plot(sorted_in, sh_from_in, color="black")
        if groups is not None:
            for id_gr, gr in enumerate(gr_fact[0]): 
                sh_in_ax.scatter(self.intercepts[gr_fact[1] == id_gr], 
                                 self.shifts[gr_fact[1] == id_gr], 
                                 c=cmap(id_gr+1), 
                                 marker=markers[id_gr],
                                 facecolors=fillstyles[id_gr], 
                                 label=gr)
            sh_in_ax.legend()
        else:
            sh_in_ax.scatter(self.intercepts, self.shifts)
        sh_in_ax.set_xlabel("Intercept")
        sh_in_ax.set_ylabel("Shift")
        
        return in_sl_fig, in_sl_ax, sh_sl_fig, sh_sl_ax, sh_in_fig, sh_sl_ax
    
    # Get zipf-mandelbrot parameters
    def get_parameters(self, slope):
        intercept = self.model_sl_in.predict(
            np.array(slope).reshape(-1, 1))[0]
        if np.sum(self.shifts) > 1e-10:
            shift = np.exp(self.model_sl_sh.predict(
                np.array(np.log(-slope)).reshape(-1, 1)))[0]
        else:
            shift = 0
        return slope, intercept, shift
    
    # Generate samples with defined slope
    def generate_samples(self, slope, sample_size, num_samples=1):
        _, intercept, shift = self.get_parameters(slope)
        types = np.arange(1, sample_size+1)
        estimated_freq = np.exp(np.log(types + shift)*slope + intercept)
        probabilities = estimated_freq / np.sum(estimated_freq)
        samples = np.array([np.where(
            np.random.multinomial(1, probabilities, sample_size) > 0)[1] + 1 
                            for _ in range(num_samples)])
        return samples
        
            
    
