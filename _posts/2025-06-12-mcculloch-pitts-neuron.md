---
layout: post
title: "McCulloch-Pitts Neuron"
author: "Sushrut"
date: 2025-06-12
math: true
categories: [deep learning, neural network]
tags: [mp-neuron, thresholding, boolean-functions, blog]
image: /assets/img/blog_posts/mp_neuron/thumbnail.png
hide_related_posts: true
toc:
  beginning: true
giscus_comments: false

scholar:
  bibliography: references.bib
  style: apa
related_publications: true
---

# Introduction

McCulloch (a neuroscientist) and Pitts (a logician) proposed a highly simplified computational model of the biological neuron in 1943, known as the **McCulloch-Pitts (MP) neuron**, in their seminal paper <d-cite key="mcculloch1943logical"></d-cite> {% cite mcculloch1943logical %}. The goal was to try and mimic the biological neuron and check if some simple functions can be implemented.

# Thresholding Logic

Consider $$d$$ binary input features represented as $$x_1$$, $$x_2$$, $$\dots$$, $$x_d$$, and a binary output $$y$$. This is shown in [Figure 1](#fig-general-mp-neuron).

<figure id="fig-general-mp-neuron">
  <div style="text-align: center;">
    <img src="/assets/img/blog_posts/mp_neuron/general_mp_neuron.png" alt="A general McCulloch-Pitts (MP) neuron." style="width:30%">
  </div>
  <figcaption style="text-align: center;">Figure 1: A general McCulloch-Pitts (MP) neuron.</figcaption>
</figure>

What this model does is something very simple. It first aggregates all the inputs, which is indicated by $$g$$, and then applies a basic threshold function $$f$$ on this aggregation. More precisely, the following is what happens. The inputs $$x_1$$, $$x_2$$, $$\dots$$, $$x_d$$ can be excitatory or inhibitory.

- If any of the inputs $$x_j$$ (where $$j\in \{1, 2, \dots, d\}$$) is inhibitory:
  - Then the output is $$y=0$$
- Else if none of the inputs $$x_j$$ are inhibitory:
  - The function $$g$$ aggregates the inputs, i.e.,

    $$
    \begin{equation}\label{eq:mp_neuron_aggregation}
    g\left(x_1, x_2, \dots, x_d\right) = g\left(\mathbf{x}\right) = \sum_{j=1}^{d} x_j,
    \end{equation}
    $$

    and the output $$y$$ is given by:

    $$
    \begin{equation}\label{eq:mp_neuron_thresholding_logic}
    y = f (g (\mathbf{x})) =
    \begin{cases}
        1, & \text{if $g(\mathbf{x}) \geq \theta$},\\
        0, & \text{if $g(\mathbf{x}) < \theta$}.
    \end{cases}
    \end{equation}
    $$

    This can be more simply written as:

    $$
    \begin{equation}\label{eq:mp_neuron_thresholding_logic_simpler}
    y = f \left(x_1, \dots, x_d\right) =
    \begin{cases}
        1, & \text{if $x_1 + \cdots + x_d \geq \theta$},\\
        0, & \text{if $x_1 + \cdots + x_d < \theta$}.
    \end{cases}
    \end{equation}
    $$

    Here, $$\theta$$ is called the **thresholding parameter**. So, the output $$y$$ is $$1$$ if the aggregation (or the sum) of all the inputs is greater than or equal to the thresholding parameter $$\theta$$, and $$0$$ otherwise.

Note that in Equation \eqref{eq:mp_neuron_thresholding_logic}, we are representing all the inputs $$x_1$$, $$x_2$$, $$\dots$$, $$x_d$$ collectively in the vector $$\mathbf{x}$$, which is given by

$$
\mathbf{x} =
\begin{bmatrix}
    x_1\\
    x_2\\
    \vdots\\
    x_d
\end{bmatrix}
$$

To understand the thresholding logic, consider the example of learning when you would like a movie given some of its features (or inputs) that are binary. You are given a data of movies. The output $$y$$ of a movie is $$1$$ if you like the it, and $$0$$ if you don't. The binary features can be

$$
\begin{align*}
x_1 &= \text{Is the director Nolan?}\\
x_2 &= \text{Is the movie Sci-Fi?}\\
x_3 &= \text{Is the IMDB rating $>8$?}\\
\vdots
\end{align*}
$$

Think of these features as favourable checks. In other words, if more of these binary features have the value $$1$$, it is more likely that you would like the movie. The thresholding parameter $$\theta$$ here models your personality. If you are someone who would like any movie, then $$\theta$$ would be lower for you. However, if you like the movie only when most of your favourable checks are satisfied, then $$\theta$$ would be higher for you.

# Implementing Boolean Functions

What it means to implement a boolean function is the following. Consider the AND boolean function. The output of this function is $$1$$ only if all the inputs are $$1$$. Else, the output is $$0$$. So, an MP neuron will be able to implement this boolean function if it exactly models this behavior.

## AND Function

Recall that the equation for an MP neuron is given by Equation \eqref{eq:mp_neuron_thresholding_logic_simpler}. Consider that there are $$3$$ boolean inputs $$x_1$$, $$x_2$$, and $$x_3$$, and all of them are excitatory. So, the MP neuron modeling the AND function should have three inputs. Hence, substituting $$d=3$$ in Equation \eqref{eq:mp_neuron_thresholding_logic_simpler}, we get

$$
y = f \left(x_1, x_2, x_3\right) =
\begin{cases}
    1, & \text{if $x_1 + x_2 + x_3 \geq \theta$},\\
    0, & \text{if $x_1 + x_2 + x_3 < \theta$}.
\end{cases}
$$

Now, the AND function outputs $$1$$ only if *all the inputs* are $$1$$. If all the outputs are $$1$$, we will have their sum as

$$
x_1 + x_2 + x_3 = 1 + 1 + 1 = 3
$$

This clearly indicates that for an MP neuron modeling the AND function with three inputs, the value of the threshold $$\theta$$ should be $$3$$. Hence, substituting $$\theta = 3$$, we get the equation of this MP neuron as

$$
y = f \left(x_1, x_2, x_3\right) =
\begin{cases}
    1, & \text{if $x_1 + x_2 + x_3 \geq 3$},\\
    0, & \text{if $x_1 + x_2 + x_3 < 3$}.
\end{cases}
$$

## OR Function

Again, recall that the equation for an MP neuron is given by Equation \eqref{eq:mp_neuron_thresholding_logic_simpler}. Consider that there are $$3$$ boolean inputs $$x_1$$, $$x_2$$, and $$x_3$$, and all of them are excitatory. So, the MP neuron modeling the OR function should have three inputs. Hence, substituting $$d=3$$ in Equation \eqref{eq:mp_neuron_thresholding_logic_simpler}, we get

$$
y = f \left(x_1, x_2, x_3\right) =
\begin{cases}
    1, & \text{if $x_1 + x_2 + x_3 \geq \theta$},\\
    0, & \text{if $x_1 + x_2 + x_3 < \theta$}.
\end{cases}
$$

Now, the OR function outputs $$1$$ only if *at least one* of the inputs is $$1$$. If this is true, then the *least value* of their sum is

$$
x_1 + x_2 + x_3 = 1
$$

This clearly indicates that for an MP neuron modeling the OR function with three inputs, the value of the threshold $$\theta$$ should be $$1$$. Hence, substituting $$\theta = 1$$, we get the equation of this MP neuron as

$$
y = f \left(x_1, x_2, x_3\right) =
\begin{cases}
    1, & \text{if $x_1 + x_2 + x_3 \geq 1$},\\
    0, & \text{if $x_1 + x_2 + x_3 < 1$}.
\end{cases}
$$

## NOT Function

For an MP neuron to model the NOT function, we will have to consider just a single input $$x_1$$ that is inhibitory. We will have two cases:

- If $$x_1 = 1$$, then $$y = 0$$. This case is sorted as this is the exact behavior of the NOT function if the input is $$1$$.
- If $$x_1 = 0$$, then using Equation \eqref{eq:mp_neuron_thresholding_logic_simpler} with $$d=1$$, we get
  
  $$
    y = f \left(x_1\right) =
    \begin{cases}
        1, & \text{if $x_1 \geq \theta$},\\
        0, & \text{if $x_1 < \theta$}.
    \end{cases}
  $$

  As we have $$x_1=0$$, and we want the output $$y$$ to be the opposite, i.e., $$y=1$$ if $$x_1 = 0$$, we can model it by assigning $$\theta = 0$$. Hence, substituting $$\theta = 0$$, we get the equation of this MP neuron in this case as

  $$
    y = f \left(x_1\right) =
    \begin{cases}
        1, & \text{if $x_1 \geq 0$},\\
        0, & \text{if $x_1 < 0$}.
    \end{cases}
  $$

Can any boolean function be represented using an MP neuron? To answer this, we will first have to understand the geometric interpretation of an MP neuron.

# Geometric Interpretation of an MP Neuron

Consider an MP neuron implementing the OR function with two inputs $$x_1$$ and $$x_2$$. The four possible inputs to the neuron are: $$(0, 0)$$, $$(0, 1)$$, $$(1, 0)$$, and $$(1, 1)$$. The output, using Equation \eqref{eq:mp_neuron_thresholding_logic_simpler} with $$d=2$$ and $$\theta = 1$$, is given by

$$
y =
\begin{cases}
    1, & \text{if $x_1 + x_2 \geq 1$},\\
    0, & \text{if $x_1 + x_2 < 1$}.
\end{cases}
$$

Clearly, we get a linear decision boundary given by the equation

$$
x_1 + x_2 = 1.
$$

Any input point lying on or in the positive half space of this boundary has an output of $$1$$, and any input point lying in the negative half space of this boundary has an output of $$0$$. [Figure 2](#or-function-2-inputs-mp-neuron) shows this.

<figure id="or-function-2-inputs-mp-neuron">
  <div style="text-align: center;">
    <img src="/assets/img/blog_posts/mp_neuron/or_function_2_inputs_mp_neuron.png" alt="An MP neuron implementing an OR function with two inputs. The green points have an output of 1 and the red point has an output of 0." style="width:50%">
  </div>
  <figcaption style="text-align: center;">Figure 2: An MP neuron implementing an OR function with two inputs. The green points have an output of 1 and the red point has an output of 0.</figcaption>
</figure>

So, a single MP neuron splits the input points into two halves using a linear boundary, or more generally a hyperplane, given by

$$
x_1 + x_2 + \cdots + x_d = \theta,
$$

where $$d$$ is the number of inputs. All inputs that produce an output of $$0$$ will be on one side of this hyperplane, and all the inputs that produce an output of $$1$$ will be on the other side of this hyperplane. In other words, a single MP neuron can be used to represent boolean functions that are *linearly separable*. Linear separability for boolean function means that there exists a line (plane) such that all inputs with an output of $$1$$ line on one side of this line (plane), and all inputs with an output of $$0$$ lie on the other side of this line (plane).

# Questions to Ponder

Are there any non-separable boolean functions? If so, can they be represented using an MP neuron? We will address these questions eventually. More precisely, we would like to address the following questions:

- What about non-boolean (say, real) inputs?
- Do we always need to hand code the threshold?
- Are all inputs equally important?
- What about functions that are not linearly separable?

# Acknowledgment

I have referred to the YouTube playlists {% cite iitm-deep-learning-playlist %} and {% cite nptel-deep-learning-playlist %} to write this blog.
