"""Plotting Helper

This module allows for quick and easy plot creation using matplotlib.

Although this module doesn't provide nearly as much flexibility as matplotlib directly, 
it does allow you to setup hassle-free plots quickly without having to type out a plethora
of matplotlib plot and formatting commands.

This module is therefore mainly intended as a time-saver and for general data exploration. 
For more flexibility and custom plot creation, refer to matplotlib directly. 

Filename: plotting_helper.py
Maintainers: David Hobson, Saruggan Thiruchelvan
Last Updated: December 21, 2021
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

##############################################################################################################
## Individual Plotting Functions
##############################################################################################################

def plot_vertical(x_values, y_values, title="", x_label="", y_label="", colour='tab:red', figsize=None, width=0.75, x_label_orientation='horizontal', plot_filepath="", save=False):
    """
    Creates a bar chart. 
    plot_vertical(x_values, y_values, title, x_label, y_label, colour, figsize, width, x_label_orientation, plot_filepath, save)
    
    x_values: list
        The x-values to plot.
    y_values: list
        The y-values to plot.
    title: str; optional (default="")
        The title for the plot.
    x_label: str: optional (default="")
        The label for the x-axis.
    y_label: str: optional (default="")
        The label for the y-axis.
    colour: str; optional (default='tab:red')
        The colour of the bars.
    figsize: (int, int); optional (default=None)
        The size of the figure. (Width x Height).
    width: float; optional (default=0.75)
        The width of each bar. Represents the percentage of available space the bar will take up.
        Must be between 0 and 1.
    x_label_orientation: str or float; optional (default='horizontal')
        The orientation of the labels on the x-axis. This can either be 'vertical' or 'horizontal' or can be a
        float indicating a degree of rotation.
    plot_filepath: str; optional (default="")
        The path to save the file to.
    save: bool; optional (default=False)
        True if the file should be saved to the specified filepath.
    """

    if figsize is None:
        fig, ax = plt.subplots() 
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # plot bar graph
    ax.bar(x_values, y_values, width, color=colour)

    # format the title, x-axis, y-axis
    ax.set_title(title)
    
    # re-orient the x-axis if applicable
    if x_label_orientation == 'vertical' or x_label_orientation == 'v':
        plt.xticks(rotation="vertical")                                # makes x-names vertical
    elif isinstance(x_label_orientation, int):
        plt.xticks(rotation=x_label_orientation)                       # makes x-names slanted at given angle
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))              # makes y-axis integer valued

    plot_filepath = plot_filepath

    if save:
        plt.savefig(plot_filepath, bbox_inches="tight")
    plt.show()

def plot_horizontal(x_values, y_values, title="", x_label="", y_label="", colour='tab:red', figsize=None, plot_filepath="", save=False):
    """
    Creates a horizontal bar chart. 
    plot_horizontal(x_values, y_values, title, x_label, y_label, colour, figsize, plot_filepath, save)
    
    x_values: list
        The x-values to plot.
    y_values: list
        The y-values to plot.
    title: str; optional (default="")
        The title for the plot.
    x_label: str: optional (default="")
        The label for the x-axis.
    y_label: str: optional (default="")
        The label for the y-axis.
    colour: str; optional (default='tab:red')
        The colour of the bars.
    figsize: (int, int); optional (default=None)
        The size of the figure. (Width x Height).
    plot_filepath: str; optional (default="")
        The path to save the file to.
    save: bool; optional (default=False)
        True if the file should be saved to the specified filepath.
    """
    
    if figsize is None:
        fig, ax = plt.subplots() 
    else:
        fig, ax = plt.subplots(figsize=figsize)

    y_positions = np.arange(len(y_values))           # set the positions along the y-axis

    # plot the bars
    ax.barh(y_positions, x_values, align='center', color=colour)

    # format the plot
    ax.set_title(title)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_values)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))          # makes x-axis integer valued

    plot_filepath = plot_filepath

    # save if desired
    if save:
        plt.savefig(plot_filepath, bbox_inches="tight")
        
    plt.show()


##############################################################################################################
## Group Functions
##############################################################################################################

def plot_groups_stacked(group_data, group_names, title="", x_label="", y_label="", colours=None, figsize=None, width=0.75, orientation='vertical', x_label_orientation='horizontal', plot_filepath="", save=False):
    """
    Create a stacked bar chart. That is, a bar chart where the data for the same x-value from different groups is stacked into the same bar. 
    plot_groups_stacked(group_data, group_names, title, x_label, y_label, colours, figsize, width, orientation, x_label_orientation, plot_filepath, save)
    
    group_data: {str: [float]}
        The data for each group.
        The keys of the dict are the x-values to be plotted.
        The values correspond to the list of y-values; one for each group. These lists must be the same length as the group_names argument.
    group_names: [str]
        The list of the group names. These are shown in the legend.
    title: str; optional (default="")
        The title for the plot.
    x_label: str: optional (default="")
        The label for the x-axis.
    y_label: str: optional (default="")
        The label for the y-axis.
    colours: [str]
        The list of bar colours for each of the groups. This must be the same length as the group_names argument.
    figsize: (int, int); optional (default=None)
        The size of the figure. (Width x Height).
    width: float; optional (default=0.75)
        The width of each bar. Only applicable to vertical bar charts. Represents the percentage of available space the bar will take up.
        Must be between 0 and 1.
    orientation: str; optional (default='vertical')
        The orientation of bars on the bar chart. Must be either 'vertical' or 'v' (for vertical bars) or 'horizontal' or'h'
        for horizontal bars.
    x_label_orientation: str or float; optional (default='horizontal')
        The orientation of the labels on the x-axis. Only applicable to vertical bar charts. This can either be 'vertical' or 'horizontal' or can be a
        float indicating the degree of rotation.
    plot_filepath: str; optional (default="")
        The path to save the file to.
    save: bool; optional (default=False)
        True if the file should be saved to the specified filepath.
    """
    
    if not isinstance(colours, list):
        print("Error: The colours argument needs to be a list, and have the same length as group_names")
        return
    elif len(colours) != len(group_names):
        print("Error: The colours needs to have the same length as group_names")
        return
    
    if orientation == 'vertical' or orientation == 'v':
        return __plot_groups_stacked_vertical(group_data, group_names, title=title, x_label=x_label, y_label=y_label, colours=colours, figsize=figsize, width=width, x_label_orientation = x_label_orientation, plot_filepath=plot_filepath, save=save)
    elif orientation == 'horizontal' or orientation == 'h':
        return __plot_groups_stacked_horizontal(group_data, group_names, title=title, x_label=x_label, y_label=y_label, colours=colours, figsize=figsize, plot_filepath=plot_filepath, save=save)
    else:
        print("Orientation can only be either 'vertical' or 'horizontal' (or 'v' or 'h' for short).")
        return

def __plot_groups_stacked_vertical(group_data, group_names, title="", x_label="", y_label="", colours=None, figsize=None, width=0.75, x_label_orientation='horizontal', plot_filepath="", save=False):
    """
    <PRIVATE> Helper function to create a stacked vertical bar chart. 
    __plot_groups_stacked_vertical(group_data, group_names, title, x_label, y_label, colours, figsize, width, x_label_orientation, plot_filepath, save)
    
    group_data: {str: [float]}
        The data for each group.
        The keys of the dict are the x-values to be plotted.
        The values correspond to the list of y-values; one for each group. These lists must be the same length as
        the group_names argument.
    group_names: [str]
        The list of the group names. These are shown in the legend.
    title: str; optional (default="")
        The title for the plot.
    x_label: str: optional (default="")
        The label for the x-axis.
    y_label: str: optional (default="")
        The label for the y-axis.
    colours: [str]
        The list of bar colours for each of the groups.
    figsize: (int, int); optional (default=None)
        The size of the figure. (Width x Height).
    width: float; optional (default=0.75)
        The width of each bar. Represents the percentage of available space the bar will take up.
        Must be between 0 and 1.
    x_label_orientation: str or float; optional (default='horizontal')
        The orientation of the labels on the x-axis. This can either be 'vertical' or 'horizontal' or can be a
        float indicating the degree of rotation.
    plot_filepath: str; optional (default="")
        The path to save the file to.
    save: bool; optional (default=False)
        True if the file should be saved to the specified filepath.
    """
    
    # get the values for the plot
    labels = group_data.keys()                        # x-values
    data = np.array(list(group_data.values()))        # y-values for each group as an array
    data_cum = data.cumsum(axis=1)                 # cumulative sum of the y-values for each group (needed to stack the bars)
    
    # choose a figure size
    if figsize is None:
        fig, ax = plt.subplots() 
    else:
        fig, ax = plt.subplots(figsize=figsize)
    
    # adjust the colours
    if not isinstance(colours, list):
        if isinstance(colours, str):
            colours = [colours for i in group_names]
        else:
            print("The colours argument needs to be a string or a list. Using the default red colour.")
            colours = ['tab:red' for i in group_names]
    
    # create and plot each bar
    for i in range(len(group_names)):
        ax.bar(labels, data[:, i], color=colours[i], bottom=data_cum[:, i] - data[:, i], label=group_names[i])
    
    # re-orient the x-axis if applicable
    if x_label_orientation == 'vertical' or x_label_orientation == 'v':
        plt.xticks(rotation="vertical")                                # makes x-names vertical
    elif isinstance(x_label_orientation, int):
        plt.xticks(rotation=x_label_orientation)                       # makes x-names slanted at given angle

    # format the plot 
    ax.set_title(title)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))          # makes y-axis integer valued
    ax.legend()

    # optionally save the figure
    if save:
        plt.savefig(plot_filepath, bbox_inches="tight")

    plt.show()

def __plot_groups_stacked_horizontal(group_data, group_names, title="", x_label="", y_label="", colours=None, figsize=None, plot_filepath='', save=False):
    """
    <PRIVATE> Helper function to create a horizontal stacked bar chart. 
    plot_groups_stacked_horizontal(group_data, group_names, title, x_label, y_label, colours, figsize, plot_filepath, save)
    
    group_data: {str: [float]}
        The data for each group.
        The keys of the dict are the x-values to be plotted.
        The values correspond to the list of y-values; one for each group. These lists must be the same length as
        the group_names argument.
    group_names: [str]
        The list of the group names. These are shown in the legend.
    title: str; optional (default="")
        The title for the plot.
    x_label: str: optional (default="")
        The label for the x-axis.
    y_label: str: optional (default="")
        The label for the y-axis.
    colours: [str]
        The list of bar colours for each of the groups. This must be the same length as the group_names argument.
    figsize: (int, int); optional (default=None)
        The size of the figure. (Width x Height).
    plot_filepath: str; optional (default="")
        The path to save the file to.
    save: bool; optional (default=False)
        True if the file should be saved to the specified filepath.
    """
    
    # get the values for the groups
    labels = list(group_data.keys())                  # x-values
    data = np.array(list(group_data.values()))        # y-values for each group as an array
    data_cum = data.cumsum(axis=1)                    # cumulative sum of the y-values for each group (needed to stack the bars)
    
    # choose a figure size
    if figsize is None:
        fig, ax = plt.subplots() 
    else:
        fig, ax = plt.subplots(figsize=figsize)
    
    # adjust the colours
    if not isinstance(colours, list):
        if isinstance(colours, str):
            colours = [colours for i in group_names]
        else:
            print("The colours argument needs to be a string or a list. Using the default red colour.")
            colours = ['tab:red' for i in group_names]
            
    # invert the axes
    ax.invert_yaxis()
    
    # format the plot
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))          # makes x-axis integer valued
    ax.set_xlim(0, np.sum(data, axis=1).max())

    # add the colours to the bars
    for i, (colname, colour) in enumerate(zip(group_names, colours)):
        
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=colour)
        xcenters = starts + widths / 2

    ax.legend(loc='best')
    
    # optionally save the figure
    if save:
        plt.savefig(plot_filepath, bbox_inches="tight")

    plt.show()

def plot_groups_clustered(group_data, group_names, title="", x_label="", y_label="", colours=None, figsize=None, plot_filepath="", save=False):
    """
    Creates a clustered bar plot. That is, a bar chart with multiple adjacent bars corresponding to data from different groups.
    plot_groups_clustered(group_data, group_names, title, x_label, y_label, colours, figsize, plot_filepath, save)
    
    group_data: {str: [float]}
        The data for each group.
        The keys of the dict are the x-values to be plotted.
        The values correspond to the list of y-values; one for each group. These lists must be the same length as the group_names argument.
    group_names: [str]
        The list of the group names. These are shown in the legend.
    title: str; optional (default="")
        The title for the plot.
    x_label: str: optional (default="")
        The label for the x-axis.
    y_label: str: optional (default="")
        The label for the y-axis.
    colours: [str]
        The list of bar colours for each of the groups. This must be the same length as the group_names argument.
    figsize: (int, int); optional (default=None)
        The size of the figure. (Width x Height).
    plot_filepath: str; optional (default="")
        The path to save the file to.
    save: bool; optional (default=False)
        True if the file should be saved to the specified filepath.
    """  
    
    if not isinstance(colours, list):
        print("Error: The colours argument needs to be a list, and have the same length as group_names")
        return
    elif len(colours) != len(group_names):
        print("Error: The colours needs to have the same length as group_names")
        return
    
    labels = group_data.keys()
    # group percentages by group as opposed to by word
    group_percentages = [ [group_data[label][i] for label in group_data] for i in range(len(group_names)) ]
    
    N = len(labels)                    
    x_values = np.arange(N)           # array of x-values where each new word starts
    width = 0.15                      # width of each individual bar

    fig = plt.figure(figsize=figsize)
    
    # plot the results
    for i in range(len(group_names)):
        # plot the results for one party
        plt.bar(x_values + i*width, group_percentages[i], width, color=colours[i], label=group_names[i])

    # set the title, y-axis, x-axis, and legend
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_values + (len(group_names) - 1)*width / 2, tuple(labels), rotation=45)
    plt.legend(loc='best')

    # optionally save the image
    if save:
        plt.savefig(plot_filepath, bbox_inches="tight")

    plt.show()
    

def plot_group_results_individually(group_results, title_template="", x_label="", y_label="", colours='tab:red', figsize=None, width=0.75, orientation='vertical', x_label_orientation='horizontal', plot_filepath_template="", save=False):
    """
    Plot multiple bar charts for different groups at once. One bar chart is created for each group.
    plot_groups_results_individually(group_results, title_template, x_label, y_label, colours, figsize, width, orientation, x_label_orientation, plot_filepath_template, save)
 
    group_results: {str: ([object], [object])}
        The data for the groups.
        The keys of the group names.
        The values are tuples containing the x- and y-values for that group (each as a list). ([x-values], [y-values]).
    title_template: str; optional (default="")
        The template for the titles of the plots. Anywhere there is an asterick (*) in the title template, that character will be replaced by the group name.
    x_label: str: optional (default="")
        The label for the x-axis.
    y_label: str: optional (default="")
        The label for the y-axis.
    colours: str or [str]
        The colour, or list of colours to be used for the plots. If one colour is specified, it will be used as the colour for all the plots. If a list is used,
        it need not have the same length as the number of groups. The colours will be cycled through each of the plots.
    figsize: (int, int); optional (default=None)
        The size of the figure. (Width x Height).
    width: float; optional (default=0.75)
        The width of each bar. Only applicable for vertical bar charts. Represents the percentage of available space the bar will take up.
        Must be between 0 and 1.
    orientation: str; optional (default='vertical')
        The orientation of bars in the bar charts. Must be either 'vertical' or 'v' (for vertical bars) or 'horizontal' or'h' for horizontal bars.
    x_label_orientation: str or float; optional (default='horizontal')
        The orientation of the labels on the x-axis. Only applicable for vertical bar charts. This can either be 'vertical' or 'horizontal' or can be a
        float indicating the degree of rotation.
    plot_filepath_template: str; optional (default="")
        The template for the filepaths of the plots. Anywhere there is an asterick (*) in the template, that character will be replaced by the group name.
    save: bool; optional (default=False)
        True if the file should be saved to the specified filepath.
    """
    
    if orientation == 'vertical' or orientation == 'v':
        return __plot_group_results_individually_vertical(group_results, title_template=title_template, x_label=x_label, y_label=y_label, colours=colours, width=width, x_label_orientation=x_label_orientation, figsize=figsize, plot_filepath_template=plot_filepath_template, save=save)
    elif orientation == 'horizontal' or orientation == 'h':
        return __plot_group_results_individually_horizontal(group_results, title_template=title_template, x_label=x_label, y_label=y_label, colours=colours, figsize=figsize, plot_filepath_template=plot_filepath_template, save=save)
    else:
        print("Orientation can only be either 'vertical' or 'horizontal' (or 'v' or 'h' for short).")
        return


def __plot_group_results_individually_vertical(group_results, title_template="", x_label="", y_label="", colours=None, figsize=None, width=0.75, x_label_orientation='horizontal', plot_filepath_template="", save=False):
    """
    <PRIVATE> Helper function to plot multiple vertical bar charts for different groups at once. 
    __plot_groups_results_individually_vertical(group_results, title_template, x_label, y_label, colours, figsize, width, x_label_orientation, plot_filepath_template, save)
    
    group_results: {str: ([object], [object])}
        The data for the groups.
        The keys of the group names.
        The values are tuples containing the x- and y-values for that group (each as a list). ([x-values], [y-values]).
    title_template: str; optional (default="")
        The template for the titles of the plots. Anywhere there is an asterick (*) in the title template, that character will be replaced by the group name.
    x_label: str: optional (default="")
        The label for the x-axis.
    y_label: str: optional (default="")
        The label for the y-axis.
    colours: str or [str]
        The colour, or list of colours to be used for the plots. If one colour is specified, it will be used as the colour for all the plots. If a list is used,
        it need not have the same length as the number of groups. The colours will be cycled through each of the plots.
    figsize: (int, int); optional (default=None)
        The size of the figure. (Width x Height).
    width: float; optional (default=0.75)
        The width of each bar. Represents the percentage of available space the bar will take up.
        Must be between 0 and 1.
    x_label_orientation: str or float; optional (default='horizontal')
        The orientation of the labels on the x-axis. This can either be 'vertical' or 'horizontal' or can be a
        float indicating the degree of rotation.
    plot_filepath_template: str; optional (default="")
        The template for the filepaths of the plots. Anywhere there is an asterick (*) in the template, that character will be replaced by the group name.
    save: bool; optional (default=False)
        True if the file should be saved to the specified filepath.
    """
    
    group_names = list(group_results.keys())
    results = list(group_results.values())
    
    if not isinstance(colours, list):
        if isinstance(colours, str):
            colours = [colours]
        else:
            print("The colours argument needs to be a string or a list. Using the default red colour.")
            colours = ['tab:red']
    
    # plot results for each group
    for i in range(len(group_results)):

        plot_vertical(x_values=results[i][0], y_values=results[i][1], title=title_template.replace('*', group_names[i]), x_label=x_label, y_label=y_label, colour=colours[i%len(colours)], figsize=figsize, width=width, x_label_orientation=x_label_orientation, plot_filepath=plot_filepath_template.replace('*', group_names[i]), save=save)
         
def __plot_group_results_individually_horizontal(group_results, title_template="", x_label="", y_label="", colours=None, figsize=None, plot_filepath_template="", save=False):
    """
    <PRIVATE> Helper function to plot multiple horizontal bar charts for different groups at once.
    __plot_groups_results_individually_horizontal(group_results, title_template, x_label, y_label, colours, figsize, plot_filepath_template, save)
    
    group_results: {str: ([object], [object])}
        The data for the groups.
        The keys of the group names.
        The values are tuples containing the x- and y-values for that group (each as a list). ([x-values], [y-values]).
    title_template: str; optional (default="")
        The template for the titles of the plots. Anywhere there is an asterick (*) in the title template, that character will be replaced by the group name.
    x_label: str: optional (default="")
        The label for the x-axis.
    y_label: str: optional (default="")
        The label for the y-axis.
    colours: str or [str]
        The colour, or list of colours to be used for the plots. If one colour is specified, it will be used as the colour for all the plots. If a list is used,
        it need not have the same length as the number of groups. The colours will be cycled through each of the plots.
    figsize: (int, int); optional (default=None)
        The size of the figure. (Width x Height).
    plot_filepath_template: str; optional (default="")
        The template for the filepaths of the plots. Anywhere there is an asterick (*) in the template, that character will be replaced by the group name.
    save: bool; optional (default=False)
        True if the file should be saved to the specified filepath.
    """
    
    group_names = list(group_results.keys())
    results = list(group_results.values())
    
    if not isinstance(colours, list):
        if isinstance(colours, str):
            colours = [colours]
        else:
            print("The colours argument needs to be a string or a list. Using the default red colour.")
            colours = ['tab:red']    

    # plot the results for each group
    for i in range(len(results)):

        plot_horizontal(x_values=results[i][1], y_values=results[i][0], title=title_template.replace('*', group_names[i]), x_label=x_label, y_label=y_label, colour=colours[i%len(colours)], figsize=figsize, plot_filepath=plot_filepath_template.replace('*', group_names[i]), save=save)