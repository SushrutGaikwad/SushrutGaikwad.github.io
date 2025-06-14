---
layout: post
title: "McCulloch-Pitts Neuron"
author: "Sushrut"
date: 2025-06-12
math: true
categories: [deep learning, neural network]
tags: [mp-neuron, thresholding, boolean-functions, blog]
image: /assets/img/blog_posts/mp_neuron/thumbnail.png
---
# Introduction

McCulloch (a neuroscientist) and Pitts (a logician) proposed a highly simplified computational model of the biological neuron in 1943, known as the **McCulloch-Pitts (MP) neuron**, in their seminal paper<a id="cite1" href="#ref1"><sup>[1]</sup></a>. The goal was to try and mimic the biological neuron and check if some simple functions can be implemented.

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
  - $$y=0$$
- If none of the inputs $$x_j$$ are inhibitory:
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












# References

<ol>
  <li id="ref1">
    W. S. McCulloch and W. Pitts, “A logical calculus of the ideas immanent in nervous activity,” <i>The Bulletin of Mathematical Biophysics</i>, vol. 5, pp. 115–133, 1943. <a href="#cite1">[↩]</a>
  </li>

  <li id="ref2">
    IIT Madras - B.S. Degree Programme, “Deep Learning - IIT Madras B.S. Degree,” YouTube Playlist, 2023. [Online]. Available: <a href="https://youtube.com/playlist?list=PLZ2ps__7DhBZVxMrSkTIcG6zZBDKUXCnM">https://youtube.com/playlist?list=PLZ2ps__7DhBZVxMrSkTIcG6zZBDKUXCnM</a>. <a href="#cite2">[↩]</a>
  </li>

  <li id="ref3">
    NPTEL-NOC IITM, “Deep Learning,” YouTube Playlist, 2019. [Online]. Available: <a href="https://youtube.com/playlist?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT">https://youtube.com/playlist?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT</a>. <a href="#cite3">[↩]</a>
  </li>
</ol>
