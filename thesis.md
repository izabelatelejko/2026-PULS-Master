\documentclass[a4paper,11pt,twoside]{report}
% THIS FILE SHOULD BE COMPILED BY pdfLaTeX

% ----------------------   PREAMBLE PART ------------------------------

% ------------------------ ENCODING & LANGUAGES ----------------------

\usepackage[utf8]{inputenc}
%\usepackage[MeX]{polski} % Not needed unless You have a name with polish symbols or sth
\usepackage[T1]{fontenc}
\usepackage[english, polish]{babel}
\usepackage[round]{natbib}
\usepackage{bm}
\usepackage{bbold}
\usepackage{algpseudocode}
\usepackage{algorithm}
\usepackage{multirow}
\usepackage[table]{xcolor} % for coloring the table
\usepackage{adjustbox}
\usepackage{booktabs}
\usepackage{diagbox}
\usepackage{subfigure}
\usepackage{dirtree}
\usepackage[toc,page]{appendix}
\usepackage{amsmath, amsfonts, amsthm, latexsym} % MOSTLY MATHEMATICAL SYMBOLS

\usepackage[final]{pdfpages} % INPUTING TITLE PDF PAGE - GENERATE IT FIRST!
%\usepackage[backend=bibtex, style=verbose-trad2]{biblatex}


\usepackage{commath} % various commands which can make writing math expressions easier --- documentation available at: https://ctan.gust.org.pl/tex-archive/macros/latex/contrib/commath/commath.pdf

\usepackage[hidelinks]{hyperref} % for hyperlinks, for example, urls, references to equations, entries in a bibliography --- hidelinks option removes rectangles around hiperlinks


% ---------------- MARGINS, INDENTATION, LINESPREAD ------------------

\usepackage[inner=20mm, outer=20mm, bindingoffset=10mm, top=25mm, bottom=25mm]{geometry} % MARGINS


\linespread{1.5}
\allowdisplaybreaks         % ALLOWS BREAKING PAGE IN MATH MODE

\usepackage{indentfirst}    % IT MAKES THE FIRST PARAGRAPH INDENTED; NOT NEEDED
\setlength{\parindent}{5mm} % WIDTH OF AN INDENTATION


%---------------- RUNNING HEAD - CHAPTER NAMES, PAGE NUMBERS ETC. -------------------

\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
% PAGINATION: LEFT ALIGNMENT ON EVEN PAGES, RIGHT ALIGNMENT ON ODD PAGES 
\fancyfoot[LE,RO]{\thepage} 
% RIGHT HEADER: zawartość \rightmark do lewego, wewnętrznego (marginesu) 
\fancyhead[LO]{\sc \nouppercase{\rightmark}}
% lewa pagina: zawartość \leftmark do prawego, wewnętrznego (marginesu) 
\fancyhead[RE]{\sc \leftmark}

\renewcommand{\chaptermark}[1]{\markboth{\thechapter.\ #1}{}}

% HEAD RULE - IT'S A LINE WHICH SEPARATES HEADER AND FOOTER FROM CONTENT
\renewcommand{\headrulewidth}{0 pt} % 0 MEANS NO RULE, 0.5 MEANS FINE RULE, THE BIGGER VALUE THE THICKER RULE


\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[LE,RO]{\thepage}
  
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0.0pt}
}



% --------------------------- CHAPTER HEADERS ---------------------

\usepackage{titlesec}
\titleformat{\chapter}
  {\normalfont\Large \bfseries}
  {\thechapter.}{1ex}{\Large}

\titleformat{\section}
  {\normalfont\large\bfseries}
  {\thesection.}{1ex}{}
\titlespacing{\section}{0pt}{30pt}{20pt} 

    
\titleformat{\subsection}
  {\normalfont \bfseries}
  {\thesubsection.}{1ex}{}


% ----------------------- TABLE OF CONTENTS SETUP ---------------------------

\def\cleardoublepage{\clearpage\if@twoside
\ifodd\c@page\else\hbox{}\thispagestyle{empty}\newpage
\if@twocolumn\hbox{}\newpage\fi\fi\fi}


% THIS MAKES DOTS IN TOC FOR CHAPTERS
\usepackage{etoolbox}
\makeatletter
\patchcmd{\l@chapter}
  {\hfil}
  {\leaders\hbox{\normalfont$\m@th\mkern \@dotsep mu\hbox{.}\mkern \@dotsep mu$}\hfill}
  {}{}
\makeatother

\usepackage{titletoc}
\makeatletter
\titlecontents{chapter}% <section-type>
  [0pt]% <left>
  {}% <above-code>
  {\bfseries \thecontentslabel.\quad}% <numbered-entry-format>
  {\bfseries}% <numberless-entry-format>
  {\bfseries\leaders\hbox{\normalfont$\m@th\mkern \@dotsep mu\hbox{.}\mkern \@dotsep mu$}\hfill\contentspage}% <filler-page-format>

\titlecontents{section}
  [1em]
  {}
  {\thecontentslabel.\quad}
  {}
  {\leaders\hbox{\normalfont$\m@th\mkern \@dotsep mu\hbox{.}\mkern \@dotsep mu$}\hfill\contentspage}

\titlecontents{subsection}
  [2em]
  {}
  {\thecontentslabel.\quad}
  {}
  {\leaders\hbox{\normalfont$\m@th\mkern \@dotsep mu\hbox{.}\mkern \@dotsep mu$}\hfill\contentspage}
\makeatother



% ---------------------- TABLES AD FIGURES NUMBERING ----------------------

\renewcommand*{\thetable}{\arabic{chapter}.\arabic{table}}
\renewcommand*{\thefigure}{\arabic{chapter}.\arabic{figure}}


% ------------- DEFINING ENVIRONMENTS FOR THEOREMS, DEFINITIONS ETC. ---------------

\makeatletter
\newtheoremstyle{definition}
{3ex}%                           % Space above
{3ex}%                           % Space below
{\upshape}%                      % Body font
{}%                              % Indent amount
{\bfseries}%                     % Theorem head font
{.}%                             % Punctuation after theorem head
{.5em}%                          % Space after theorem head, ' ', or \newline
{\thmname{#1}\thmnumber{ #2}\thmnote{ (#3)}}
\makeatother

\theoremstyle{definition}
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{example}[theorem]{Example}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{remark}[theorem]{Remark}

% --------------------- END OF PREAMBLE PART (MOSTLY) --------------------------





% -------------------------- USER SETTINGS ---------------------------

\newcommand{\tytul}{Klasyfikacja dla danych typu Positive Unlabeled przy zmianie rozkładu a priori}
\renewcommand{\title}{Classification of Positive Unlabeled data under label shift}
\newcommand{\type}{Master} % Master OR Engineer
\newcommand{\supervisor}{prof. dr hab. Jan Mielniczuk} % TITLE AND NAME OF THE SUPERVISOR



\begin{document}
\sloppy
\selectlanguage{english}

\includepdf[pages=-]{titlepage} % THIS INPUTS THE TITLE PAGE

\null\thispagestyle{empty}\newpage

% ------------------ PAGE WITH SIGNATURES --------------------------------

%\thispagestyle{empty}\newpage
%\null
%
%\vfill
%
%\begin{center}
%\begin{tabular}[t]{ccc}
%............................................. & \hspace*{100pt} & .............................................\\
%supervisor's signature & \hspace*{100pt} & author's signature
%\end{tabular}
%\end{center}
%


% ---------------------------- ABSTRACTS -----------------------------

{  \fontsize{12}{14} \selectfont
\begin{abstract}

\begin{center}
\title
\end{center}

Many real-world machine learning applications involve settings where only partially labeled data are available. A particular binary classification scenario, known as Positive Unlabeled (PU) learning, arises when only labeled positive and unlabeled observations are available for training. Another challenge frequently encountered in practice is the phenomenon known as label shift, where the class proportions differ between the training and target data. Although several methods have been proposed to adapt classifiers to label shift, most of them were developed for fully supervised learning and require adaptation to the PU setting.

This thesis investigates label shift adaptation in case-control PU learning. We evaluate an existing approach and propose novel techniques based on threshold adaptation of classification functions and posterior adjustment using the Expectation-Maximization algorithm. The methods are compared within two PU learning frameworks: non-negative PU learning and density ratio based learning. We also investigate several approaches to class prior estimation, which is necessary for determining the degree of label shift.

The proposed methods are evaluated on both synthetic and real-world datasets under a variety of label shift configurations. As part of the evaluation, we introduce a procedure for simulating label shift in PU data. The experimental results indicate that label shift adaptation methods do not necessarily improve classification performance in PU learning. However, additional analysis suggests that classification accuracy is often relatively insensitive to threshold modifications. Overall, the study provides a systematic comparison of label shift adaptation techniques in PU learning and identifies the directions for future research.

\noindent \textbf{Keywords:}  machine learning, semi-supervised learning, positive-unlabeled data, label shift, class prior estimation
\end{abstract}
}

\null\thispagestyle{empty}\newpage


{\selectlanguage{polish} \fontsize{12}{14}\selectfont
\begin{abstract}

\begin{center}
\tytul
\end{center}

Wiele rzeczywistych zastosowań uczenia maszynowego opiera się na danych, dla których dostępne są jedynie częściowe etykiety. Szczególnym przypadkiem problemu binarnej klasyfikacji jest uczenie z danych pozytywnych i nieoznaczonych (ang. Positive-Unlabeled Learning), w którym do trenowania modelu dostępne są wyłącznie otykietowane obserwacje pozytywne oraz obserwacje nieoznaczone. Innym wyzwaniem często występującym w praktyce jest zjawisko przesunięcia rozkładu a priori (ang. label shift), polegające na różnicy w proporcji klas pomiędzy zbiorem treningowym a zbiorem docelowym. Chociaż zaproponowano wiele metod adaptacji klasyfikatorów do tego zjawiska, większość z nich została opracowana dla problemów uczenia nadzorowanego i wymaga dostosowania do uczenia danych typu Positive Unlabeled (PU).

Niniejsza praca poświęcona jest problemowi adaptacji modeli PU do zjawiska label shift w scenariuszu case-control. Analizujemy już istniejącą metodę oraz proponujemy nowe podejścia oparte na adaptacji progu decyzyjnego klasyfikatora oraz modyfikacji predykcji a posteriori z wykorzystaniem algorytmu Expectation-Maximization. Metody porównywane są w ramach dwóch podejść do uczenia PU: nieujemnego uczenia PU (nnPU) oraz uczenia opartego na estymacji ilorazu gęstości (DRPU). Dodatkowo badamy różne metody estymacji a priori, niezbędne do stwierdzenia przesunięcia rozkładu.

Proponowane metody zostały ocenione zarówno na danych syntetycznych, jak i rzeczywistych, dla różnych konfiguracji zmiany a priori. Na potrzeby badań opracowano również procedurę symulacji zjawiska label shift dla danych PU. Uzyskane wyniki wskazują, że metody adaptacji do label shift nie zawsze prowadzą do poprawy skuteczności klasyfikacji w problemach PU. Dodatkowa analiza sugeruje, że dokładność klasyfikacji jest często względnie niewrażliwa na modyfikacje progu decyzyjnego. Badanie dostarcza systematycznego porównania metod adaptacji do label shift w uczeniu PU oraz wskazuje kierunki dalszych badań.

\noindent \textbf{Słowa kluczowe:} uczenie maszynowe, uczenie półnadzorowane, uczenie z danych typu Positive Unlabeled, label shift, estymacja a priori
\end{abstract}
}


%% --------------------------- DECLARATIONS ------------------------------------
%
%%
%%	IT IS NECESSARY OT ATTACH FILLED-OUT AUTORSHIP DEECLRATION. SCAN (IN PDF FORMAT) NEEDS TO BE PLACED IN scans FOLDER AND IT SHOULD BE CALLED, FOR EXAMPLE, DECLARATION_OF_AUTORSHIP.PDF. IF THE FILENAME OR FILEPATH IS DIFFERENT, THE FILEPATH IN THE NEXT COMMAND HAS TO BE ADJUSTED ACCORDINGLY.
%%
%%	command attacging the declarations of autorship
%%
%\includepdf[pages=-]{scans/declaration-of-autorship}
%\null\thispagestyle{empty}\newpage
%
%% optional declaration
%%
%%	command attaching the declaataration on granting a license
%%
%\includepdf[pages=-]{scans/declaration-on-granting-a-license}
%%
%%	.tex corresponding to the above PDF files are present in the 3. declarations folder 
%
\null\thispagestyle{empty}\newpage
% ------------------- TABLE OF CONTENTS ---------------------
% \selectlanguage{english} - for English
\pagenumbering{gobble}
\tableofcontents
\thispagestyle{empty}
\newpage % IF YOU HAVE EVEN QUANTITY OD PAGES OF TOC, THEN REMOVE IT OR ADD \null\newpage FOR DOUBLE BLANK PAGE BEFORE INTRODUCTION


% -------------------- THE BODY OF THE THESIS --------------------------------

\null\thispagestyle{empty}\newpage
\pagestyle{fancy}
\pagenumbering{arabic}
\setcounter{page}{11}


\chapter{Introduction}

In many real-world scenarios, access to the fully labeled dataset is often limited. We only observe a subset of positive samples, and the remaining observations are unlabeled. This setting requires the development of classifiers that can be trained without explicitly labeled negative samples. Positive Unlabeled (PU) learning addresses this challenge using a positive set $X_P$ and an unlabeled set $X_U$. 

In the case-control scenario, most classifiers are constructed under the assumption that the data to be classified follow the same distribution of predictors as in the training data. However, in real-world applications, datasets often exhibit class imbalances and varying proportions of positive and negative samples, making this assumption unrealistic. This phenomenon of changes in the class prior probabilities between the training and test datasets, called label shift, affects the quality of classifiers as the posterior probabilities modeled during training directly depend on the prior probabilities. The objective of this work is to adapt existing classifiers to effectively handle such label shifts.

An additional complexity arises from the fact that while the training priors may sometimes be known, the prior probabilities during testing are inherently unknown and must be estimated. Accurately estimating these priors is critical for enabling classifiers to effectively adjust to the label shift and make reliable predictions in the test data. Without a precise prior estimation, corrections for the shift may be flawed, resulting in diminished classification performance. This study presents several methods for prior estimation that can be applied to PU data.

The only known approach to address label shift in PU learning was proposed by \cite{nakajima2022}. Their method involves estimating the new posterior probabilities using density ratio estimation and adapting the PU classifier to account for the shifted priors. This work aims to evaluate their approach and introduce the novel methods based on classification threshold adaptation and Expectation-Maximization-based posterior adjustment for addressing label shift in Positive Unlabeled data, with the objective of comparing and analyzing various solutions.

\chapter{Theory} \label{theory}

In this chapter, the theoretical foundations relevant to the study are presented. The concept of label shift is introduced, along with its implications for classification models. Subsequently, the framework of Positive Unlabeled (PU) learning is presented, including the derivation of the PU risk estimator and the challenges it faces, such as ensuring non-negativity through appropriate adjustments. Finally, the combination of PU learning and label shift is addressed, with a focus on the density ratio method.

\section{Label Shift}

We consider the anti-causal learning setting \citep{schoelkopf2012}, where the labels $Y$ are considered to be the cause of the features $X$. This assumption implies that the class-conditional distributions $p(x|y)$ remain invariant, which means that they are the same during the training and testing phases. However, the marginal distribution of labels $p(y)$ may vary between the train and test datasets. This shift in the class distribution is known as label shift and is a very common issue in real-world scenarios, particularly in applications like medical diagnosis. For instance, in disease detection, where diseases (the target variable) cause symptoms (the predictors), label shift often arises because the training dataset is constructed to include all registered cases of sick patients. This methodology, driven by the rarity of certain diseases, leads to an increased proportion of sick patients in the training data compared to the general population. 

Label shift has been studied extensively in recent years. Early work such as \citep{saerens2002} introduced Maximum Likelihood Label Shift (MLLS), an iterative and EM-based method to estimate shifted label distributions. More recent studies focus on efficient estimation methods that work well in high-dimensional settings. \citet{lipton2018} introduced Black Box Shift Estimation (BBSE), which uses predictions from pre-trained classifiers to estimate label shift via moment-matching on confusion matrices. Advantage of BBSE is its flexibility, as it does not require explicit calibration of classifiers and is compatible with modern deep learning models. However, \citet{garg2020} demonstrated that BBSE might lose statistical efficiency due to coarse calibration, and MLLS outperforms BBSE under certain conditions.

In the multi-class classification setting, let $X \in \mathcal{X} = \mathbb{R}^d$ represent the input data and $Y \in \mathcal{Y} = \{1, 2, \dots, K\}$ denote the class labels, where $K$ is the total number of classes. Let $P$ and $P'$ represent the distributions of the training and test data, respectively. The probability density function $p$ of the training data can be expressed as:

$$
p(x) = \sum_{k=1}^K P(Y=k) p(x \mid Y=k) = \sum_{k=1}^K p(k) p(x | k),
$$

where $P(Y = k) = P_{P}(Y = k) = p(k)$ is the prior probability of class $k$, and $p(x \mid Y=k) = p(x | k)$ is the class-conditional density for class $k$. When label shift occurs, the class priors differ between the training and test datasets, resulting in a modified distribution $p'$ in test data:

$$
p'(x) = \sum_{k=1}^K p'(k) p(x | k),
$$

where $p'(k) = P_{P'}(Y = k)$ is the prior probability of class $k$ in the test data. The priors satisfy $\sum_{k=1}^K p(k) = \sum_{k=1}^K p'(k) = 1$. Label shift assumes that the overall data distribution changes due to differences in priors, while the class-conditional densities $p(x \mid Y=k)$ remain unchanged.

Classifiers trained with a specific class prior may fail on test data with different priors. The posterior probability on which classification rules are based, depends on priors:

$$
p(k | x) \propto p(x | k) p(k),
$$

where $p(k | x) = p(Y=k | x)$.

When priors change, posterior probabilities are affected, and classification decisions constructed without taking this into account may no longer be optimal. 

\subsection{Maximum Likelihood Label Shift} \label{mlls}

\cite{saerens2002} proposed an iterative procedure to adjust classifier outputs to new prior probabilities without retraining the model.

By Bayes' theorem, the within-class densities for training data are the following:

$$
p(x | k) = \frac{p(k | x)p(x)}{p(k)}.
$$

For a new dataset with new priors, the analogous formula is:

$$
p'(x | k) = \frac{p'(k | x)p'(x)}{p'(k)}.
$$

Assuming within-class densities do not change, we have the following equality:

$$
p(x | k) = p'(x | k).
$$

Let us define $f(x) = \frac{p(x)}{p'(x)}$, and derive the formula for posterior probability in test data from the equality above:

\begin{equation} \label{eq:ls-post}
p'(k | x) = f(x) \frac{p'(k)}{p(k)} p(k | x).
\end{equation}

Although we do not know the posterior probabilities $p'(k | x)$, by summing them over all classes $\sum_{l=1}^{K}p'(l | x) = 1$, we are able to calculate $f(x)$:

$$
f(x) = \left[ \sum_{l=1}^{K} \frac{p'(l)}{p(l)} p(l | x) \right]^{-1}.
$$

Substituting $f(x)$ back to \ref{eq:ls-post} results in the explicit relation between train and test posteriors:

\begin{equation} \label{eq:mlls}
    p'(k | x) = \frac{\frac{p'(k)}{p(k)} p(k | x)}{\sum_{l=1}^{K} \frac{p'(l)}{p(l)} p(l | x)}.
\end{equation}

During training, we obtain the estimators of posterior probabilities $\hat{p}(k | x_k)$ from a classification model such as Naive Bayes or logistic regression. Moreover, prior probabilities can be easily estimated as $\hat{p}(k) = \frac{N_t^k}{N_t}$, where $N_k$ is the number of training samples in class $k$, and $N_t$ is the total number of training samples. However, to use the adjustment formula \ref{eq:ls-post}, the prior estimates for the test data are required. To address this missing information, \cite{saerens2002} used the Expectation-Maximization (EM) algorithm \citep{dempster1977}, which iteratively estimates the priors and adjusts the classifier. 

The initial guesses of the test priors are simply the empirical train priors:

$$
\hat{p'}^{(0)}(k) = \hat{p}(k), k \in \{1, 2, \dots, K\}.
$$

The EM steps at iteration $s$ are as follows:

\begin{itemize}
    \item E-step: Calculating the expected value of train posteriors:
    $$
    \hat{p'}^{(s)}(k | x_k) = \frac{\frac{\hat{p}'^{(s)}(k)}{\hat{p}(k)} \hat{p}(k | x_k)}{\sum_j \frac{\hat{p}'^{(s)}(k_j)}{\hat{p}(k_j)} \hat{p}(k_j | x_k)}.
    $$
    This formula is a result of Equation \ref{eq:mlls}.
    \item M-step: Updating the estimates of train priors to maximize the likelihood of the test data.
    $$
    \hat{p'}^{(s+1)}(k) = \frac{1}{N} \sum_{j=1}^{N} \hat{p}'^{(s)}(k | x_j).
    $$
\end{itemize}

\subsection{Black Box Shift Estimation}

\citet{lipton2018} introduced Black Box Shift Estimation (BBSE), aiming at estimating the ratio of label priors $p'(y)/p(y)$ between the test and training distributions. BBSE utilizes a pre-trained classifier to estimate moments of the label distributions and the associated covariance matrices. Using the estimated ratios $w_i = p'(y_i)/p(y_i)$ as importance weights, BBSE solves the Importance-Weighted Empirical Risk Minimization (WERM) problem to obtain a classifier that is adapted to label shift.

BBSE method operates under the following assumptions:

\begin{itemize}
    \item[A.1] \textbf{Label shift assumption:} $p(x|y) = p'(x|y), \quad \forall x \in \mathcal{X}, y \in \mathcal{Y}$.  
    This fundamental assumption, already highlighted in previous section follows from the anti-casual learning setting.

    \item[A.2] \textbf{Support condition:} For every $y \in \mathcal{Y}$ with $p'(y) > 0$, it holds that $p(y) > 0$. In other words: $\text{supp}(P) \subseteq \text{supp}(P')$.
    This ensures that the training data contain observations from all classes observed in test data.

    \item[A.3] \textbf{Confusion matrix invertibility:} A classifier $g: \mathcal{X} \rightarrow \mathcal{Y}$ is considered for which the expected confusion matrix $C_p(g) \in \mathbb{R}^{|\mathcal{Y}| \times |\mathcal{Y}|} $ is invertible, where:  
    $$
    [C_p(g)]_{ij} = p(g(x) = i \mid y = j).
    $$  
    This condition ensures that the expected classifier's predictions for each class are linearly independent.
\end{itemize}

In order to derive the formula for test priors, let us first state the following lemma:

\begin{lemma}
Let $g: \mathcal{X} \rightarrow \mathcal{Y}$ and denote $\hat{y} = g(x)$ Under the label shift assumption (A.1), the conditional prediction distributions remain invariant:
$$
p'(\hat{y}|y) = p(\hat{y}|y).
$$
\end{lemma}

The above equality is a simple consequence of the fact that the distribution of $\hat{y} = \hat{y}(x)$ given $y$ is a function of densities $p(x \mid y) = p'(x \mid y)$.

From the law of total probability and the above lemma we obtain:

\begin{equation} \label{eq:bbse}
  p'(\hat{y}) = \sum_{y \in \mathcal{Y}} p'(\hat{y}|y)p'(y) = \sum_{y \in \mathcal{Y}} p(\hat{y}|y)p'(y) = \sum_{y \in \mathcal{Y}} p(\hat{y}, y) \frac{p'(y)}{p(y)}.  
\end{equation}

Equation \ref{eq:bbse} highlights that the test distribution of predictions $p'(\hat{y})$ depends on the training joint distribution $p(\hat{y}, y)$ and the ratio of  priors $p'(y)/p(y)$. Next, we define in vector representation the class priors $\bm{\mu}_y, \bm{\mu'}_y \in \mathbb{R}^{|\mathcal{Y}|}$ of the distributions $p$, and $p'$, along with their estimates:

\[
\begin{array}{ll}
\left[ \bm{\mu}_y \right]_i = p(y = i), & \left[ \bm{\mu'}_y \right]_i = p'(y = i), \\[10pt]
\left[ \bm{\mu}_{\hat{y}} \right]_i = p(g(x) = i), & \left[ \bm{\mu'}_{\hat{y}} \right]_i = p'(g(x) = i), \\[10pt]
\left[ \bm{\hat{\mu}}_{\hat{y}} \right]_i = \frac{\sum_{j} \mathbb{1} \left\{ g(x_j) = i \right\} }{n}, & \left[ \bm{\hat{\mu}'}_{\hat{y}} \right]_i = \frac{\sum_{j} \mathbb{1} \left\{ g(x'_j) = i \right\} }{n}.
\end{array}
\]

The ratio of the priors $\bm{w} \in \mathbb{R}^{|\mathcal{Y}|}$ can be expressed as:

$$
\left[ w \right]_i = \frac{p'(y=i)}{p(y=i)}.
$$

Additionally, we define covariance matrix $C_{\hat{y}, y} \in \mathbb{R}^{|\mathcal{Y}| \times |\mathcal{Y}|}$, and its empirical counterpart, which describe the relationship between the true and predicted labels:

\[
\begin{aligned}
\left[ C_{\hat{y}, y} \right]_{ij} &= p(g(x) = i, y = j), \\
% \left[ C_{\hat{y} \mid y} \right]_{ij} &= p(g(x) = i \mid y = j), \\
\left[ \hat{C}_{\hat{y}, y} \right]_{ij} &= \frac{1}{n} \sum_{l} \mathbb{1} \left\{ g(x_l) = i \text{ and } y_l = j \right\}.
\end{aligned}
\]

Using the above definitions, we can rewrite Equation \ref{eq:bbse} in matrix form:

$$
\bm{\mu}_{\hat{y}} = C_{\hat{y}, y} \bm{w}.
$$

The weights $\bm{w}$, representing the ratio of priors, can then be estimated as:

$$
\hat{\bm{w}} = C^{-1}_{\hat{y}, y} \bm{\mu}_{\hat{y}}.
$$

Once the weights $\hat{\bm{w}}$ are estimated, they are incorporated into the Weighted Empirical Risk Minimization (WERM) framework \citep{vogel2020}, an adaptation of the classical Empirical Risk Minimization (ERM) \citep{devroye1996}. WERM modifies the standard ERM objective by applying importance weights to adjust for shifts in the data distribution, such as label shift. Standard ERM minimizes the expected loss over the training samples, defined as:

$$
\mathcal{L}(g) = \mathbb{E}_{(x, y) \sim P}[\ell(g(x), y)].
$$

For the binary case, the concepts of loss and risk will be discussed more thoroughly in the next section.

Under label shift, the importance weights $\hat{w}_i = p'(y_i)/p(y_i)$ modify this objective to account for the shifted test distribution:

$$
\mathcal{L}_{\text{WERM}}(g) = \mathbb{E}_{(x, y) \sim P}\left[{w}_i \ell(g(x_i), y_i)\right].
$$

The classifier is then trained by minimizing the empirical approximation of this importance-weighted loss:

\begin{equation} \label{eq:bbse_g}
    \Tilde{g} = \arg \min_g \frac{1}{n} \sum_{i=1}^n \hat{w}_i \ell(g(x_i), y_i).
\end{equation}

In practice, estimated weights $\hat{w}$ may occasionally contain negative values due to noise in the estimates or small sample sizes. To address this, clipping is often applied to ensure positivity, thus preventing potential issues with unbounded losses during optimization. BBSE has been shown to be consistent under assumptions A.1-A.3 and provides interpretable error bounds. Its scalability and compatibility with deep learning models make it a popular choice for label shift estimation.


\section{Positive Unlabeled Learning}

In this section, various approaches to Positive-Unlabeled (PU) learning are presented. The objective of PU learning is to train a binary classifier using only positive and unlabeled data, without direct access to explicitly labeled negative samples. We stress that from now on we consider the binary case due to the specificity of the PU problem. Figure \ref{fig:pu_data_ex} presents a comparison of a standard Positive-Negative (PN) learning and Positive-Unlabeled (PU) learning problem on sample data.

\begin{figure}[h] 
    \centering
    \includegraphics[width=0.9\textwidth]{2. thesis/img/pu_data_ex.png}
    \vspace{-1em}
    \caption{Comparison of standard Positive-Negative (PN) learning and Positive-Unlabeled (PU) learning problem. In the PU setting, the unlabeled set, shown in grey, consists of all negative samples and a subset of positive samples.}  \label{fig:pu_data_ex}
\end{figure}

In the PU learning framework, the classification task involves two variables: $X \in \mathcal{X} = \mathbb{R}^d$, representing the input feature space, and $Y \in \mathcal{Y} = \{-1, +1\}$, the binary labels, where $+1$ denotes the positive class and $-1$ denotes the negative class.

The marginal distributions are defined as follows: $p_p(x) = p(x \mid Y = +1)$ represents the conditional distribution of the positive class, $p_n(x) = p(x \mid Y = -1)$ represents the conditional distribution of the negative class, and $p(x)$ denotes the overall distribution of the unlabeled data. In the PU setting, and case-control scenario, only positive samples ($X_p$) and unlabeled samples ($X_u$) are observed. Specifically, $X_p = \{x_i^p\}_{i=1}^{n_p}$ is the set of $n_p$ positive samples drawn from the $p_p(x)$ distribution, and $X_u = \{x_i^u\}_{i=1}^{n_u}$ is the set of $n_u$ unlabeled samples drawn from the $p(x)$ distribution.

From now on, the class prior probabilities are denoted as $\pi_p = p(Y = +1)$ for the positive class and $\pi_n = 1 - \pi_p = p(Y = -1)$ for the negative class. In most cases, $\pi_p$ is assumed to be known or estimated directly from the positive and unlabeled data.

We define the fundamental concepts of classification functions, loss functions, and risk functions, formulated for binary classification, to establish the mathematical framework for Positive Unlabeled learning.

\begin{definition}[Classification function]
In binary classification, the classification function $g \colon \mathbb{R}^d \to \mathbb{R}$ maps input features $x \in \mathbb{R}^d$ to a score, where the predicted class is determined by the sign of the classification function:
$$
I(x) = 
\operatorname{sign}(g(x)) =
\begin{cases} 
+1, & \text{if } g(x) > 0, \\ 
-1, & \text{if } g(x) \leq 0.
\end{cases}
$$
\end{definition}

\begin{definition}[Loss function]
The loss function is a mapping $l \colon \mathbb{R} \times \{-1, +1\} \to \mathbb{R}^+$. The value of $l(t, y)$ measures the cost of predicting the class based on the score $t$ for the true label $y$. The popular loss functions used in PU learning include:
\begin{itemize}
    \item Zero-one loss: $l(t, y) = \operatorname{sign}(ty)$,
    \item Hinge loss: $l(t, y) = \max(0, 1 - yt)$,
    \item Logistic loss: $l(t, y) = \log(1 + e^{-yt})$,
    \item Squared loss: $l(t, y) = (t - y)^2$.
\end{itemize}
\end{definition}

\begin{definition}[Risk function]
The risk function $R$ of classification function $g$ is the expected loss over the data distribution:
$$
R(g) = \mathbb{E}[l(g(X), Y)].
$$
The empirical risk is defined as the sample-based approximation of $R(g)$:
$$
\hat{R}(g) = \frac{1}{N} \sum_{i=1}^N l(g(x_i), y_i).
$$
\end{definition}

\subsection{Two-Step Techniques for PU Learning}

The two-step techniques in PU learning aim to identify reliable negative examples in the first step. In the second step, these reliable negatives are combined with the labeled positive samples to train a supervised classifier. Some approaches introduce an optional third step to select the best-performing classifier obtained in step two. These methods typically rely on the assumptions of separability and smoothness, where positive and negative examples are distinguishable from each other and similar within their respective classes. While various methods can be applied at each step, the literature usually recommends specific combinations.

\subsubsection{S-EM}

One method, the Spy-EM (S-EM) algorithm proposed by \citet{liu2003}, combines the spy methodology with the Expectation-Maximization Naive Bayes (EM-NB). The spy method begins by selecting a small fraction of the labeled positive examples as "spies" (typically 10\%), which are temporarily treated as part of the unlabeled set. A Naive Bayes classifier is then trained on the remaining positive examples and the entire unlabeled set, where the unlabeled examples are initially assumed to belong to the negative class. All positive samples are assigned a posterior probability of belonging to the positive class, equal to 1, while the observations in the unlabeled set are initially assigned a posterior probability of 0. The EM algorithm iteratively updates the posterior probabilities, and after several iterations, these probabilities converge.

The reliable negative observations are identified as those examples in the unlabeled set for which the posterior probabilities are lower than a certain threshold. This threshold is defined based on the posterior probabilities of the "spies", as we know that they belong to positive class. Once the reliable negatives are identified, the "spies" are returned to the positive set. As a result, the dataset is divided into three sets: the positive set, the reliable negative set, and the remaining unlabeled set.

In the second step, a new Naive Bayes classifier is trained using these three sets. The positive set is assigned the positive label, the reliable negative set is assigned the negative label, and the remaining unlabeled set is left unassigned initially. Then again, EM algorithm is applied iteratively in this setting, updating the posterior probabilities at each iteration until convergence.

\subsubsection{ROC-SVM}

Another method, Rocchio-SVM (ROC-SVM), proposed by \citet{li2003}, combines the Rocchio algorithm \citep{rocchio71} with an iterative Support Vector Machine (SVM) classifier. In the first step, the Rocchio algorithm is utilized to distinguish between positive and negative classes by constructing prototypes for both positive and unlabeled data. These prototypes are derived as weighted differences of the mean vectors of feature vectors representing observations for the respective classes. Unlabeled examples that are closer to the negative prototype than to the positive prototype are chosen as reliable negatives.

Originally, the Rocchio algorithm was designed for textual data (documents), where feature vectors were formed using a term frequency-inverse document frequency (tf-idf) weighting scheme \citep{tf-idf}. In this representation, each word $w_i$ is assigned a value $q_i = \text{tf}_i \cdot \text{idf}_i$, where $\text{tf}_i$ is the number of times word $w_i$ occurs in a document, and $\text{idf}_i$ measures how unique $w_i$ is within the entire set of documents. The feature vector for a document is a vector of tf-idf values for all words in the document. For non-text data, feature vectors can be derived directly from the numerical or categorical attributes of the observations. 

In the second step, the reliable negatives identified in the first step, along with the labeled positive examples, are used to train an SVM classifier. The SVM iteratively updates the reliable negatives set by incorporating unlabeled samples that are classified as negative by the current SVM model. A new SVM is then trained using the updated definition of the negative set. The iterative process continues until the fraction of positive observations classified as negative exceeds 5\%.

ROC-SVM performs particularly well when the data are linearly separable, as both Rocchio and SVM are linear classifiers.

\subsubsection{MCLS}

The Maximum Margin Clustering with LS-SVM (MCLS) method \citep{chaudhari2012} combines clustering with a non-linear least squares SVM (LS-SVM).

In the first step, k-means clustering is applied to divide the dataset into clusters. Each cluster is labeled as positive or negative based on the proportion of positive examples it contains. From the clusters identified as negative, some observations that are the farthest from the positive centroid are selected as reliable negatives, as these points have a higher likelihood of belonging to the negative class. 

In the second step, the algorithm trains a classifier using the positive and reliable negative examples with least squares SVM \citep{suykens1999}. The training process optimizes the classifier parameters, such as weights $w$ and bias $b$, and iteratively updates the labels of the unlabeled data based on the classifier's predictions. An important aspect of the method is maintaining the class balance ratio during this process to avoid trivial solutions. It is controlled by the bias parameter $b$, which is adjusted to ensure this maintenance, and that positive examples are consistently labeled correctly.

The iterative process continues until the labels of the unlabeled data stabilize and the last obtained classifier is used as the final model. Experiments conducted in \citep{chaudhari2012} demonstrated that MCLS outperforms several other methods, particularly in datasets with small number of positive observations.

\subsection{Unbiased and Non-Negative PU Learning} \label{sec:upu}

In this section methods for learning from PU data that are based on unbiased and non-negative risk estimators are explored \citep{kiryo2017}. 

In the standard binary classification problem, where we have access to both positive and negative samples, the training process often aims to minimize the risk function of the following form:

$$
{R}_{pn}(g) = \pi_p {R}_{p}^{+}(g) + \pi_n {R}_{n}^{-}(g).
$$

This risk can be estimated empirically as:

$$
\hat{R}_{pn}(g) = \pi_p \hat{R}_{p}^{+}(g) + \pi_n \hat{R}_{n}^{-}(g),
$$

where $\hat{R}_{p}^{+}(g) = \frac{1}{n_p} \sum_{i=1}^{n_p} l(g(x_i^p), +1)$, and $\hat{R}_{n}^{-}(g) = \frac{1}{n_n} \sum_{i=1}^{n_n} l(g(x_i^n), -1)$.

However, in the PU setting, we cannot directly calculate $\hat{R}_{n}^{-}(g)$. Using the relationship $\pi_n p_n(x) = p(x) - \pi_p p_p(x)$, the risk for the negative class can be expressed as:

$$
\pi_n {R}_{n}^{-}(g) = {R}_{u}^{-}(g) - \pi_p {R}_{p}^{-}(g).
$$

This results in the following risk function, adjusted to PU data:

\begin{equation} \label{upu}
{R}_{pu}(g) = \pi_p {R}_{p}^{+}(g) + {R}_{u}^{-}(g) - \pi_p {R}_{p}^{-}(g).    
\end{equation}

An empirical estimator for this risk function, based solely on the sets $X_p$ and $X_u$, is given by:

\begin{equation} \label{upu-empirical}
\hat{R}_{pu}(g) = \pi_p \hat{R}_{p}^{+}(g) + \hat{R}_{u}^{-}(g) - \pi_p \hat{R}_{p}^{-}(g),  
\end{equation}

where $\hat{R}_{u}^{-}(g) = \frac{1}{n_u} \sum_{i=1}^{n_u} l(g(x_i^u), -1)$, and $\hat{R}_{p}^{-}(g) = \frac{1}{n_p} \sum_{i=1}^{n_p} l(g(x_i^p), -1)$.
 
As shown in \citep*{NIPS2014} and \citep*{plessis2015}, risk estimator in Equation \ref{upu} is consistent when the loss function satisfies the symmetry condition: $l(t, +1) + l(t, -1) = 1$ (this implies that the loss function must be non-convex) or when the loss function is convex and satisfies the linear-odd condition: $l(t, +1) - l(t, -1) = -t$. Due to its unbiasedness, Equation \ref{upu} is referred to as the unbiased PU (uPU) risk estimator.

We will focus on logistic and sigmoid loss functions further on. Let $\eta(x)=p(Y=+1\mid x)$ denote the posterior probability of the positive class. For logistic function, which is convex and satisfies the linear-odd condition, we have:

$$
\arg\min_{g \in \mathcal{G}} R(g) = \log(\frac{\eta(x)}{1 - \eta(x)}),
$$

where $\mathcal{G} = \{g: \mathcal{X} \rightarrow \mathbb{R}\}$.
  
For sigmoid loss, which satisfies the symmetry condition, we have:

$$
\arg\min_{g \in \mathcal{G}_M} R(g) = M \text{sign}(\eta(x) - \frac{1}{2}),
$$

where $\mathcal{G}_M = \{g : \|g\|_\infty < M\}$.

\cite{kiryo2017} identified a limitation of the unbiased PU risk estimator: it tends to overfit, as the estimated risk can become negative during training. To address this, the researchers proposed the modification to formula in \ref{upu}, motivated by the fact that ${R}_{u}^{-}(g) - \pi_p {R}_{p}^{-}(g) = \pi_n {R}_{n}^{-}(g) \geq 0$. If the estimation yields the negative value of risk $R_u^-(g)-\pi_pR_p^-(g)$, it is replaced with $0$. The corrected risk estimator, known as the non-negative PU (nnPU) risk estimator, is given by:

\begin{equation} \label{nnpu}
\Tilde{R}_{pu}(g) = \pi_p \hat{R}_{p}^{+}(g) + \max\left\{0, \hat{R}_{u}^{-}(g) - \pi_p \hat{R}_{p}^{-}(g)\right\}.
\end{equation}

The modified risk estimator stabilizes the training process by preventing overfitting and ensures non-negative risk values. Experiments in \cite{kiryo2017} demonstrated that this approach leads to improved stability and performance in PU learning tasks.

\subsection{Other Methods}

Rebalancing methods proposed by \citet{elkan2001} address PU learning by weighting data to achieve desired classification thresholds without altering the classifier. These methods adjust the target threshold by utilizing the ratio of class priors. Similarly, Rank Pruning (RP), introduced by \citet{northcutt2022}, operates within the framework of classification with noisy labels, aiming to improve robustness by identifying and removing uncertain labels.

The Generative PU (GenPU) \citep{hou2018} method utilizes Generative Adversarial Networks (GANs) to model both positive and negative data distributions in PU learning. GenPU employs two generators -- one for positive and one for negative samples, and three discriminators to differentiate between real positive examples, real negative examples, and the positive or negative labels of unlabeled data. GenPU generates synthetic data that enables semi-supervised training based on positive, unlabeled and generated negative sets, improving performance on various benchmarks.

Variational PU (VPU) \citep{chen2020} utilizes a variational loss function to directly optimize the divergence between the classifier and the ideal Bayesian classifier, bypassing the need for class prior estimation. The method incorporates a MixUp-based consistency regularization, which interpolates between positive and unlabeled samples.

\section{PU Learning under Label Shift}

Non-negative PU algorithm relies on assumption that the class prior probabilities are known in advance and remain the same for the training and test data. However, this assumption is often unrealistic in practical scenarios. Training samples are frequently collected in ways that do not reflect the true proportions of positive and negative classes. For example, in rare disease detection, datasets often include all known cases of affected patients, leading to an increased fraction of the positive samples. The class proportions may also fluctuate over time, either temporarily or permanently, due to external factors. For example, during an epidemic, the fraction of sick individuals in the population temporarily increases compared to non-epidemic periods.

Even when the class priors for the training data are available, the priors for the test data are not necessarily known, particularly during the training phase. These challenges necessitate methods that not only estimate the class priors but also adapt dynamically to label shift.

In this section, we focus on approach proposed by \cite{nakajima2022} that uses density ratio estimation \citep*{sugiyama2012} to account for label shift. This method eliminates the need for prior knowledge of test class priors and ensure robust performance under varying distributions. Density ratio estimation leverages the relationship between the marginal and class-conditional densities to estimate class prior probabilities and adjust classifiers accordingly. 

The density ratio between the positive class-conditional density and the marginal input density, is defined as:

$$
r^*(x) = \frac{p_p(x)}{p(x)}.
$$

Using Bayes' rule, the posterior probability for the positive class can be expressed in terms of the density ratio:

\begin{equation} \label{eq:DRPU}
    p(Y=+1 | x) = \pi_p r^*(x).
\end{equation}

From this equation, we can observe that the training process revolves around estimating the density ratio $r^*(x)$, and it does not depend on prior knowledge of $\pi_p$. This independence from class priors during training mitigates the problem of propagating prior estimation errors in training phase. However, the class prior $\pi_p$ will still be needed during classification to compute posterior probabilities.

\subsection{Density Ratio Estimation}

In this section, we discuss the estimation of the density ratio $r^*$. As described in \citet{sugiyama2012}, Density Ratio Estimation (DRE) directly computes the ratio of two densities without requiring separate estimation of the numerator and denominator. The Bregman divergence \citep{bregman1967}, an extension of the Euclidean distance is used as a measure of similarity between the true density ratio and the proposed estimation.

\begin{definition}[Bregman Divergence]
Let $f \colon [0, \infty) \to \mathbb{R}$ be a differentiable and strictly convex function, called the generator function. The Bregman divergence associated with $f$, from $u^*$ to $u$, is defined as:
$$
BR_f(u^* \| u) = f(u^*) - f(u) - f'(u)(u^* - u),
$$
where $f'(u)$ is the derivative of $f$. The term $f(u) + f'(u)(u^* - u)$ represents the value of the first-order Taylor expansion of $f$ around $u$, evaluated at $u^*$.
\end{definition}

This divergence evaluates the difference between the value of $f$ at $u^*$ and its linear approximation from $u$. An illustrative example of Bregman divergence is shown in Figure ~\ref{fig:bregman_example}.

\begin{figure} 
    \centering
    \hspace{3cm}
    \includegraphics[width=0.7\linewidth]{2. thesis/img/bregman_ex.png} 
    \caption{The illustration of Bregman divergence for generator function $f(x)=x^2$, calculated at points $u=1$ and $u^*=2$.}\label{fig:bregman_example}
    \label{fig:enter-label}
\end{figure}

In \cite{nakajima2022} the density ratio $r^*(x)$, is estimated by minimizing the Bregman divergence between $r^*(x)$ and the estimated model $r(x)$. Using the generator function $f$, the divergence between these functions can be expressed as:

\begin{equation} \label{eq:bregman-int}
    BR_f(r^* \| r) = \int \left[ f(r^*(x)) - f(r(x)) - f'(r(x))(r^*(x) - r(x)) \right] p(x) \, dx.
\end{equation}

Expanding Equation~\ref{eq:bregman-int}, the Bregman divergence can be written as the sum of four integrals:

$$
BR_f(r^* \| r) = \int f(r^*(x))p(x)dx - \int f(r(x))p(x)dx - \int f'(r(x))r^*(x) p(x)dx + \int f'(r(x))r(x) p(x)dx.
$$

Here, the term $\int f(r^*(x))p(x)dx$ is constant with respect to $r$, since it depends only on the true density ratio $r^*(x)$. Therefore, it can be ignored during optimization:

$$
\int f(r^*(x))p(x)dx = \text{const}.
$$

The other terms can be reinterpreted using expectations over the positive and unlabeled data distributions:

$$
\int f(r(x))p(x)dx = \mathbb{E}_U[f(r(X))],
$$

$$
\int f'(r(x))r^*(x) p(x)dx = \int f'(r(x)) \frac{p_p(x)}{p(x)} p(x)dx = \int f'(r(x)) p_p(x)dx = \mathbb{E}_P[-f'(r(X))],
$$

$$
\int f'(r(x))r(x) p(x)dx = \mathbb{E}_U[f'(r(X))r(X)].
$$

Substituting these terms into Equation \ref{eq:bregman-int}, the Bregman divergence becomes:

$$
BR_f(r^* \| r) = \mathbb{E}_P[-f'(r(X))] + \mathbb{E}_U[f'(r(X))r(X) - f(r(X))] + \text{const}.    
$$

For optimization purposes, the constant term can be omitted, and we define the objective function $\mathcal{L}_f(r)$ as follows:

\begin{equation} \label{eq:bregman-e}
    \mathcal{L}_f(r) = \mathbb{E}_P[-f'(r(x))] + \mathbb{E}_U[f'(r(x))r(x) - f(r(x))].
\end{equation}

In order to further analyze the properties of function $f$, we define two auxiliary functions:

\begin{align*}
f^*(t) &= t f'(t) - f(t), \\
\mathfrak{F}(t) &= f^*(t) - f^*(0).
\end{align*}

For $t \geq 0$, taking the derivative of $f^*$:

$$
(f^*)'(t) = f'(t) + tf''(t) - f'(t) = tf''(t) > 0.
$$

Since $f$ is strictly convex, $f''(t) > 0$, and therefore $f^*$ is increasing for $t \geq 0$. Consequently, we have:

$$
\mathfrak{F}(t) \geq 0 \quad \text{for } t \geq 0.
$$

Given that $r(x) \geq 0$, for $\alpha \leq \pi_p$ we derive:

\begin{equation} \label{eq:dr0}
\mathbb{E}_U[\mathfrak{F}(r(x))] - \alpha \mathbb{E}_P[\mathfrak{F}(r(x))] \geq
\mathbb{E}_U[\mathfrak{F}(r(x))] - \pi_p \mathbb{E}_P[\mathfrak{F}(r(x))] = 
(1 - \pi_p) \mathbb{E}_N[\mathfrak{F}(r(x))] \geq 0.
\end{equation}

We can rewrite Equation \ref{eq:bregman-e} as:

\begin{align*}
\mathcal{L}_f(r) = \mathbb{E}_P[-f'(r(x))] + \mathbb{E}_U[\mathfrak{F}(r(x)) + f^*(0)] \\
= \mathbb{E}_P[-f'(r(x))] + \alpha \mathfrak{F}(r(x)) + \mathbb{E}_U[\mathfrak{F}(r(x))] - \alpha \mathbb{E}_P[\mathfrak{F}(r(x))] + f^*(0).
\end{align*}

Similarly to the approach in Equation~\ref{nnpu}, we consider only the positive values of the term $\mathbb{E}_U[\mathfrak{F}(r(x))] - \alpha \mathbb{E}_P[\mathfrak{F}(r(x))]$, as we shown in \ref{eq:dr0} that it should be non-negative.

\begin{equation} \label{drpu}
\Tilde{\mathcal{L}}_f(r) = \mathbb{E}_P[-f'(r(x))] + \alpha \mathfrak{F}(r(x)) + \max\left\{0, \mathbb{E}_U[\mathfrak{F}(r(x))] - \alpha \mathbb{E}_P[\mathfrak{F}(r(x))]\right\} + f^*(0).    
\end{equation}

We can treat $\alpha$ as a hyperparameter and tune it to minimize the empirical estimation of the objective function $\mathcal{L}_f(r)$, which is derived from Equation~\ref{eq:bregman-e}, which is independent of $\alpha$. This tuning is performed using validation datasets. Let us stress that assumption $\alpha \leq \pi_p$ is crucial as it ensures non-negativity of the term $\mathbb{E}_U[\mathfrak{F}(r(x))] - \alpha \mathbb{E}_P[\mathfrak{F}(r(x))]$.

The optimal density ratio $r^*(x)$ is estimated by minimizing this objective:

$$
\hat{r}^*(x) = \arg\min_{r} \Tilde{\mathcal{L}}_f(r).
$$

\subsection{Cost-Sensitive Classification}

In PU learning, adapting to label shift can be formulated as a cost-sensitive classification problem. In this section, we describe an approach for PU classification, based on cost-sensitive binary classification with the density ratio estimation (DRPU).

Revisiting Equation \ref{eq:DRPU}, the posterior probability for the positive class is expressed as:

$$
p(Y=+1 \mid x) = \pi_p \frac{p_p(x)}{p(x)} = \pi_p r^*(x).
$$

Thus, the optimal solution of the Bregman divergence minimization, $r = r^*$, yields a Bayes-optimal classifier by thresholding $p(Y=+1 \mid x) = \frac{1}{2}$. However, under label shift, the Bayes-optimal threshold might not classify optimally due to the change in class priors. To address this, \citet{nakajima2022} improved the classifier by incorporating cost-sensitive learning

For arbitrary false-positive cost parameter $c \in (0, 1)$, cost-sensitive classification is defined as minimizing the following risk \citep{elkan2001, scott2012}:

$$
R_{\pi, c}(g) = (1-c)\pi_p {R}_{p}^{+}(g) + c\pi_n {R}_{n}^{-}(g).
$$

\citet{charoenphakdee2019} showed that classification under class prior shift can be formulated as cost-sensitive classification. Let $\pi' \in (0, 1)$ represent the class prior of the test distribution. Then, $R_{\pi', 1/2}(g) \propto R_{\pi, c}(g)$ with:

$$
c = \frac{\pi(1-\pi')}{\pi(1-\pi') + \pi'(1-\pi)}.
$$

The Bayes-optimal risk is defined as:

$$
R^*_{\pi, c} = \inf_{g \in \mathcal{F}} R_{\pi, c}(g),
$$

where $\mathcal{F}$ is the set of all measurable functions from $\mathbb{R}^d$ to $\mathbb{R}$. The difference $R_{\pi, c}(g) - R^*_{\pi, c}$ is referred to as the excess risk for $R_{\pi, c}$. Let us define the classification function $h_{c}: \mathbb{R^d} \rightarrow \mathbb{R}$ as:

\begin{equation} \label{eq:hc}
    h_{c} = \pi r - c.
\end{equation}

\cite{nakajima2022} demonstrated that the excess risk of the function $h_{c}$ is bounded by the Bregman divergence of the density ratio estimation, $BR_f(r^* \| r)$. However, when transferring to shifted test data, these bounds may not hold directly.

For the test distribution, the classification risk is defined as:

$$
R_{\pi', c'}(g) = (1-c')\pi_p' {R}_{p}^{+}(g) + c'\pi_n' {R}_{n}^{-}(g).
$$

where $c'$ is the test-time false-positive cost.

\cite{nakajima2022} proved that by adapting threshold $c$ of $h_c$ in Equation \ref{eq:hc} to $c_0$, defined as:

$$
c_0 = \frac{c'\pi(1-\pi')}{c'\pi(1-\pi') + (1-c')\pi'(1-\pi)}.
$$

the classification risk is still bounded by the Bregman divergence. Therefore, DRPU first estimates $r^*$ as $r$, as described in the previous section, then estimates the class priors (covered in section \ref{pi_drpu}), and finally constructs the classifier $h: \mathbb{R^d} \rightarrow \mathcal{Y}$, as $h = \text{sign}(h_{c_0}) = \text{sign}(\hat{\pi} r - \hat{c_0})$.

\subsection{DRPU Algorithm}

Algorithm \ref{alg:drpu} summarizes the complete DRPU methodology for PU learning under label shift.

\begin{algorithm} 
\caption{DRPU} \label{alg:drpu}
\begin{algorithmic}[1] 
\Require Training datasets $(X_P, X_U)$, test dataset $X_U'$
\Ensure Classifier $h : \mathbb{R}^d \rightarrow \{-1, +1\}$
\State Split $(X_P, X_U)$ into training set $(X_P^{\text{tr}}, X_U^{\text{tr}})$ and validation set $(X_P^{\text{val}}, X_U^{\text{val}})$
\While{no stopping criterion has been met}
    \State Optimize $r$ with $(X_P^{\text{tr}}, X_U^{\text{tr}})$ by minimizing $\Tilde{\mathcal{L}}_f$ \Comment{Formula \ref{drpu}}
\EndWhile
\State Estimate $\hat{\pi}$ with $r$ and $(X_P^{\text{val}}, X_U^{\text{val}})$
\State Preserve the list of intervals $\{\Theta_i\}_{i=0}^{n_P}$
\State Estimate $\hat{\pi}'$ with $r$, $X_U'$, and $\{\Theta_i\}_{i=0}^{n_P}$
\State Determine $\hat{c}_0 = \frac{c_0 \hat{\pi}(1 - \hat{\pi}')}{(1 - c_0)(1 - \hat{\pi})\hat{\pi}' + c_0 \hat{\pi}(1 - \hat{\pi}')}$ 
\State \Return $h = \text{sign}(\hat{\pi}r - \hat{c}_0)$
\end{algorithmic}
\end{algorithm}

\section{Prior Estimation for PU Data}

As the label priors for the test data are unknown and, in many cases, must also be estimated for the training data, it becomes necessary to infer them from positive and unlabeled sets to enable the classification of shifted PU data. This section discusses approaches for prior estimation, including kernel mean embedding based methods (KM1 and KM2), as well as a method based on density ratio estimation.

\subsection{Kernel Mean Embedding}

\cite{ramaswamy2016} introduced the KM1 and KM2 prior estimators. The estimate of $\pi$ is obtained by considering transformation of $\mathcal{X}$ to $\phi(\mathcal{X})$, a mapping into a sufficiently rich space, called Reproducing Kernel Hilbert Space (RKHS), which ensures that $\mathbb{E}_{X \sim P} \phi(X) = \mathbb{E}_{X \sim Q} \phi(X)$ implies $P=Q$ (universality property).

\subsection{DR-based Method} \label{pi_drpu}

\cite{nakajima2022} proposed the approach based on density ratio estimation to estimate both training and test class priors.  
Given a density ratio estimator $r$, the priors are obtained as:

\begin{equation} \label{eq:drpu_prior}
\begin{aligned}
\hat{\pi}(r) = \inf_{h \in \mathcal{H}_{r}} \frac{\hat{P}(h)}{\hat{P}_+(h)}, \\
\hat{\pi}'(r) = \inf_{h \in \mathcal{H}_{r}} \frac{\hat{P}'(h)}{\hat{P}_+(h)},
\end{aligned}
\end{equation} 

where the empirical probabilities are defined as:

\begin{equation} \label{eq:drpu_prior_exp}
\begin{aligned}
\hat{P}(h) = \hat{\mathbb{E}}_U[\mathbb{1}\left\{ h(X)=+1 \right\}], \\
\hat{P}'(h) = \hat{\mathbb{E}}_{U'}[\mathbb{1}\left\{ h(X)=+1 \right\}], \\
\hat{P}_+(h) = \hat{\mathbb{E}}_P[\mathbb{1}\left\{ h(X)=+1 \right\}]. \\
\end{aligned}
\end{equation}

The infimum in Equation~\ref{eq:drpu_prior} is used because
\[
\frac{P(h)}{P_+(h)}
= \frac{\pi P_+(h) + (1-\pi)P_-(h)}{P_+(h)}
\geq \pi,
\]
which shows that the ratio provides an upper bound on the true class prior $\pi$.
Ideally, when $P_-(h)=0$, the ratio yields the true prior.

The hypothesis space $\mathcal{H}_r$ is defined as:

$$
\mathcal{H}_{r} = \left\{ h: \mathbb{R}^d \rightarrow \left\{\pm1\right\} \mid \exists \theta \in \mathbb{R}, h(x) = \text{sign}(r(x)-\theta) \wedge \hat{P}_+(h) \ge \bar{\gamma} \right\},
$$

where $\bar{\gamma}$ is a constant determined by the number of positive and unlabeled samples.

Observing that for $h = \text{sign}(r(X) - \theta)$, the values of $\hat{P}_+(h)$ remain constant and are equal to $\tfrac{i}{n_p}$
for all $\theta \in \Theta_i$, where $\{\Theta_i\}_{i=1}^{n_p}$ is the set of consecutive intervals, to find the infimum in Equation \ref{eq:drpu_prior}, it is sufficient to limit $\mathcal{H}_r$ to a discrete set of at most $n_p$ candidate classifiers:

$$
\Big\{ h_i: \mathbb{R}^d \to \{\pm1\} \,\Big|\, \Theta_i=(\theta_i; \theta_{i+1}), \
h_i(x) = \text{sign}(r(x)-\theta_i), \
\hat{P}_+(h_i) \ge \bar{\gamma} \Big\}.
$$

It is worth emphasizing that this estimation requires first computing the density ratio estimator $r$.

% \subsection{Decision Tree Induction ???}
% Article: Estimating the Class Prior in Positive and Unlabeled Data Through Decision Tree Induction
% Authors (Jessa Bekker, Jesse Davis)


\chapter{Methodology} \label{chap:methodology}

In this chapter, we present the methodology for handling label-shifted PU data. The proposed approaches combine the methods described in Chapter~\ref{theory}, introduce a novel threshold adjustment technique, and include both the nnPU and DRPU learning models. We consider three groups of methods: methods that modify the decision rule by adjusting the classification threshold, methods that directly adjust posterior probabilities to account for label shift, such as MLLS, and lastly, methods that retrain the model on a mixture of the source and target unlabeled sets.

In all experiments, we assume that the class prior in the training (source) unlabeled set is known, while the class prior in the test (target) dataset is unknown and must be estimated. This assumption may be unrealistic in a standard Positive Unlabeled setting, where the empirical prior probability of the unlabeled distribution is typically unknown and should also be estimated using the same techniques applied to the target data. However, since extending the setup to include estimation of the training prior would not introduce additional methodological contribution, we simplify the experimental design by omitting this initial estimation step. Instead, we use the empirical training prior derived as the fraction of positive obsevations in unlabeled set, and apply prior estimation methods only to infer the target prior. This simplification allow us for more comprehensive evaluation of the models and the impact of training procedure, as we do not propagate the prior estimation error at the train time. The exception from applying the known value of prior at train time is the third group of methods, based on retraining models. For these methods we ignore the source unlabeled set, and utilize the target set for training. Since the target prior is unknown, it has first to be estimated, before proceeding with training models, and the same estimate is subsequently used at test time.

\section{Threshold Adjustment Methods} \label{sec:ta}

In this section, we present a novel approach to modify the decision rule of existing classification function $g$, such as those obtained in uPU or nnPU, by adjusting the underlying threshold used for classification. We transform the scores of the classification function $g$ using sigmoid function to obtain a function $h: \mathbb{R}^d \rightarrow [0, 1]$, which models the posterior probability $P_{X \sim P}(Y = +1 \mid X)$:

$$
h(x) = S(g(x)) = \frac{1}{1 + e^{-g(x)}}.
$$

We denote $P_{X \sim P} = P_P, P_P(Y = +1) = \pi, P_{P'}(Y = +1) = \pi'$, and define the odds for the train distribution $P$, and test distribution $P'$:

$$
OD(X) = \frac{P_P(Y = +1 \mid X)}{P_P(Y = -1 \mid X)}, \quad
\Tilde{OD}(X) = \frac{P_{P'}(Y = +1 \mid X)}{P_{P'}(Y = -1 \mid X)}.
$$

Using Bayes' Theorem and the label shift assumption that within-classes distributions are the same for train and test data, we derive:

\begin{equation} \label{eq:od}
\begin{aligned}
\Tilde{OD}(X) 
&= \frac{P_{P'}(Y = +1 \mid X)}{P_{P'}(Y = -1 \mid X)} \\
&= \frac{P_{P'}(X \mid Y = +1)}{P_{P'}(X \mid Y = -1)} \frac{\pi'}{1 - \pi'} \\
&= \frac{P_{P}(X \mid Y = +1)}{P_{P}(X \mid Y = -1)} \frac{\pi'}{1 - \pi'} \\
&= \frac{P_{P}(Y = +1 \mid X)}{P_{P}(Y = -1 \mid X)} \frac{1 - \pi}{\pi} \frac{\pi'}{1 - \pi'} \\
&= OD(X) \cdot \frac{1 - \pi}{\pi} \frac{\pi'}{1 - \pi'}.
\end{aligned}
\end{equation}

In Bayesian classification, an observation is labeled as positive if posterior probability is greater than $1/2$, which in terms of odds is equivalent to $OD(X) > 1$. Therefore, for the shifted data we should classify as positive when $\Tilde{OD}(X) > 1$. Using Equation \ref{eq:od}, this inequality can be expressed in terms of train distribution odds:

$$
\Tilde{OD}(X) > 1 \equiv OD(X) > \frac{\pi}{1 - \pi} \frac{1 - \pi'}{\pi'}.
$$

The empirical counterpart of the above formula for identifying positive examples is:

\begin{equation} \label{eq:tm}
    \frac{h}{1 - h} > \frac{\pi}{1 - \pi} \frac{1 - \pi'}{\pi'}.
\end{equation}

By rearranging the terms in the above inequality, we can isolate $h$:

\begin{equation} \label{eq:tm-h}
    h > \frac{\pi (1 - \pi')}{\pi + \pi' -2\pi \pi'}.
\end{equation}

Inequality \ref{eq:tm-h} provides the classification boundary for shifted PU data with prior $\pi'$, based on the classification function trained on data with class prior $\pi$. Therefore, this threshold adjustment enables standard PU learning methods to adapt to label shift without requiring retraining. Table~\ref{tab:threshold_adjustment} presents sample adjusted thresholds calculated over a grid of nine values for training and test class priors.

\begin{table}[H]
  \centering
  \begin{tabular}{|c|ccccccccc|}
    \hline
    \diagbox[width=2.5em,height=2.5em]{$\pi$}{\raisebox{0.8ex}{$\pi'$}} & 0.1 & 0.2 & 0.3 & 0.4 & 0.5 & 0.6 & 0.7 & 0.8 & 0.9 \\
    \hline
    0.1 & \cellcolor{gray!20}\textbf{0.50} & 0.31 & 0.21 & 0.14 & 0.10 & 0.07 & 0.05 & 0.03 & 0.01 \\
    0.2 & 0.69 & \cellcolor{gray!20}\textbf{0.50} & 0.37 & 0.27 & 0.20 & 0.14 & 0.10 & 0.06 & 0.03 \\
    0.3 & 0.79 & 0.63 & \cellcolor{gray!20}\textbf{0.50} & 0.39 & 0.30 & 0.22 & 0.16 & 0.10 & 0.05 \\
    0.4 & 0.86 & 0.73 & 0.61 & \cellcolor{gray!20}\textbf{0.50} & 0.40 & 0.31 & 0.22 & 0.14 & 0.07 \\
    0.5 & 0.90 & 0.80 & 0.70 & 0.60 & \cellcolor{gray!20}\textbf{0.50} & 0.40 & 0.30 & 0.20 & 0.10 \\
    0.6 & 0.93 & 0.86 & 0.78 & 0.69 & 0.60 & \cellcolor{gray!20}\textbf{0.50} & 0.39 & 0.27 & 0.14 \\
    0.7 & 0.95 & 0.90 & 0.84 & 0.78 & 0.70 & 0.61 & \cellcolor{gray!20}\textbf{0.50} & 0.37 & 0.21 \\
    0.8 & 0.97 & 0.94 & 0.90 & 0.86 & 0.80 & 0.73 & 0.63 & \cellcolor{gray!20}\textbf{0.50} & 0.31 \\
    0.9 & 0.99 & 0.97 & 0.95 & 0.93 & 0.90 & 0.86 & 0.79 & 0.69 & \cellcolor{gray!20}\textbf{0.50} \\
    \hline
  \end{tabular}
  \caption{Adjusted model's thresholds for shifted data, for different combinations of training prior $\pi$ (rows) and test prior $\pi'$ (columns). These thresholds are derived using Equation~\ref{eq:tm-h}.}
  \label{tab:threshold_adjustment}
\end{table}

\subsection{Illustration on Positive Negative Data} \label{sec:pn_data_ex}

To illustrate and validate the effectiveness of this method, we conducted a simple experiment on synthetic binary 10-dimensional Gaussian data, where the prior probability in the test data $\pi'$ is shifted with respect to the prior in the training dataset $\pi$. Both the training and test datasets were sampled according to the following distributions:

$$
p_p \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{10}), \quad 
p_n \sim \mathcal{N}(0.5 \cdot \mathbf{1}, \mathbf{I}_{10}), 
$$
$$
p_{\text{train}} = \pi p_p + (1-\pi) p_n, \quad
p_{\text{test}} = \pi' p_p + (1-\pi') p_n.
$$

The values of $\pi$ and $\pi'$ were varied across multiple configurations, and all labels were available during training (standard PN learning setting). For simplicity, we assume that the class proportion in the test data is known. In practice this value should be estimated, but prior estimation is not the focus of this illustrative example of label shift adaptation. For each training dataset, a classification model was trained for 25 epochs using a Multi-Layer Perceptron (MLP) architecture with four hidden layers. The model outputs estimates of posterior probabilities and classifies an observation as positive if the estimate exceeds the threshold $0.5$. The trained model was then evaluated on the test dataset with a shifted prior. We analyze the classification accuracy as a function of the decision threshold by evaluating the model over a grid of threshold values. This allows us to assess whether the standard threshold of $0.5$ remains appropriate under label shift. Each experimental configuration was repeated 10 times, and the results were averaged.

Figure~\ref{fig:thres_adap_ex} presents the aggregated results for four label shift configurations. It is evident that the standard threshold of $0.5$ is not optimal for shifted datasets. In all cases, the adjusted threshold computed according to Formula~\ref{eq:tm-h} yields substantially better performance, often close to or matching the maximum accuracy. This demonstrates that threshold adaptation can significantly improve classification performance under label shift. This methodology will be further applied in the main experiments on PU data in this work.

\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{2. thesis/img/threshold_adaptation_example.png}
    \caption{An illustration of threshold adaptation method on PN data under label shift. Datasets are sampled from Gaussian distribution ($n=5,000$). The graphs present accuracy as a function of threshold. The gray line marks the standard threshold $0.5$, while the green line marks the shifted threshold that adapts to label shift.}
    \label{fig:thres_adap_ex}
\end{figure}
 
\section{Posterior Adjustment Methods} \label{posterior-adjust}

In this group of methods, we combine both learning objectives, nnPU and DRPU, with the EM-based Maximum Likelihood Label Shift (MLLS) procedure discussed in Section~\ref{mlls}. Unlike threshold adjustment strategies, MLLS directly modifies posterior probabilities to account for changes in class priors between the source and target distributions.

For the binary case, Equation~\ref{eq:mlls} simplifies to

\begin{equation}
p'(Y=+1 \mid x)
=
\frac{\frac{\pi'}{\pi} \, p(Y=+1 \mid x)}
{\frac{\pi'}{\pi} \, p(Y=+1 \mid x)
+
\frac{1-\pi'}{1-\pi} \, \left(1 - p(Y=+1 \mid x)\right)}.
\label{eq:mlls_binary}
\end{equation}

Here, $p(Y=+1 \mid x)$ denotes the posterior probability under the source distribution. In our setup, it is estimated by the nnPU model, denoted by $u(x)$, or the DRPU model, denoted by $z(x)$, both trained on the source data.

A subtle but important difference between these two models is that the ranges of their output scores differ. The nnPU model produces scores in $[0,1]$, which can be directly interpreted as posterior probabilities. On the contrary, DRPU produces density ratio scores that lie in $[0,\infty)$. These scores need to be multiplied by the source prior $\pi$ in order to transform into posterior probabilities. However, this transformation does not guarantee that the obtained values will end up in $[0, 1]$ interval. Consequently, additional clamping for DRPU scores may be required before applying the MLLS update formula.

Substituting the model output scores into Equation~\ref{eq:mlls_binary}, we obtain the following update formulas.

For nnPU:
\begin{equation}
\hat{p}'^{(s)}(Y=+1 \mid x)
=
\frac{
\frac{\hat{\pi}'^{(s)}}{\pi} \, u(x)
}{
\frac{\hat{\pi}'^{(s)}}{\pi} \, u(x)
+
\frac{1-\hat{\pi}'^{(s)}}{1-\pi} \, \left(1 - u(x)\right)
}.
\label{eq:mlls_binary2}
\end{equation}

For DRPU:
\begin{equation}
\hat{p}'^{(s)}(Y=+1 \mid x)
=
\frac{
\frac{\hat{\pi}'^{(s)}}{\pi} \, \min(\pi z(x), 1)
}{
\frac{\hat{\pi}'^{(s)}}{\pi} \, \min(\pi z(x), 1)
+
\frac{1-\hat{\pi}'^{(s)}}{1-\pi} \, \left(1 - \min(\pi z(x), 1)\right)
}.
\label{eq:mlls_binary3}
\end{equation}

The target prior $\pi'$ is estimated using the Expectation-Maximization (EM) algorithm. The procedure is initialized with $\hat{\pi'}^{(0)} = \pi$ and iterates between the following steps:

\begin{itemize}
    \item E-step: Compute adjusted posteriors for all target samples using Equations~\ref{eq:mlls_binary2} or~\ref{eq:mlls_binary3}, depending on the model.
    \item M-step: Update the target prior estimate as the empirical mean of the adjusted posteriors:
    \begin{equation}
    \hat{\pi}'^{(s+1)} = \frac{1}{N} \sum_{i=1}^{N} \hat{p}'^{(s)}(Y=+1 \mid x_i).
    \end{equation}
\end{itemize}

The iterations continue until convergence condition is met:

$$
|\hat{\pi}'^{(s+1)} - \hat{\pi}'^{(s)}| < \varepsilon,
$$

or until a predefined maximum number of iterations is reached.

Unlike threshold adjustment methods, MLLS provides both the adjusted posterior probabilities $\hat{p}'^{(s)}(Y=+1 \mid x)$, which can subsequently be used with the standard decision rule $\hat{p}'^{(s)}(Y=+1 \mid x) > 0.5$, and an estimate of the target class prior $\hat{\pi}'^{(s)}$. 

It is important to note that the EM algorithm does not guarantee global convergence, and its performance may depend on the initialization and model quality.

\section{Target Retraining Methods} \label{target-retrain}

The methods discussed in the previous two sections do not require model retraining in order to adapt to label shift. In those approaches, the model is trained once on the positive set and the unlabeled set drawn from the source distribution, which may contain different class proportions than the target distribution. The adaptation procedures are based on the model scores produced by this fixed classifier.

The final approach we consider for handling label shifted PU data abandons the assumption that training must be completed before target data become available. Instead, we completely ignore the unlabeled set coming from the source distribution $X_U$ and replace it with the unlabeled set that represents the target distribution $X_{U'}$. Consequently, training is performed on the mixture $X_P + X_{U'}$, rather than on the classical combination $X_P + X_U$. Once the batch of target data is received, the model is retrained. Essentially, the same data is used for model training and generating predictions. 

Although this approach may appear the simplest in its concept, it holds a few limitations as well. In particular, the absence of a pre-trained model can be problematic in applications requiring fast predictions. When the target dataset is not large enough, the quality of the classifier may decrease. Moreover, if the model architecture is complex, retraining may incur substantial computational cost.
 
This approach also poses problems in terms of how to conduct a fair comparison with the previously discussed methods. The performance of retrained models depends directly on the size and quality of the target unlabeled data, which may either advantage or disadvantage this procedure. Ideally, we would aim to use the same amounts of data in the source and target unlabeled sets. At the same time, the size of the dataset used for testing should always remain the same. This is not possible to achieve in practice, as for given label frequency and fixed size of train dataset, the size of unlabeled set depends on the class prior. Therefore, in the synthetic data experiments, we enforce equal total sample sizes for the source and target datasets, we set $n_p + n_u = n_{u'}$. As a result, during retraining we use $n_p + n_{u'} > n_p + n_u$ observations, which is a slightly bigger training set compared to the source-trained models. On the other hand, under our assumption of a known source prior and an unknown target prior, the retraining procedure requires estimating the target prior before training. In our experiments, we use the KM2 estimator. Consequently, methods that rely on the known source prior during training benefit from not requiring prior estimation at model training time, whereas retraining methods, which ignore the source unlabeled set, must estimate the prior first. In short, it is not possible to ensure perfectly fair experimental conditions when comparing methods from different groups. However, we design the experimental setup to approximate such conditions as closely as possible - models in retraining methods are favored by having access to a larger training sample, while at the same time being challenged by the need to estimate the class prior before the training process.

\section{Inference for Label-Shifted PU Data}

In this section, we present an overview of the methods for handling label shift in PU learning. We work with training sets $X_P$ and $X_U$, and a label-shifted unlabeled test set $X_U'$. We assume that the test priors $\pi'_p$ and $\pi'_n = 1 - \pi'_p$ are unknown and consider the scenario where training priors are known. Our goal is to evaluate both the effectiveness of prior estimation methods and the performance of PU learning models when combined with specific prior estimation techniques. 

\subsection{Learning methodologies}

In this work, we evaluate the performance of two PU learning methodologies introduced in the previous chapter: the non-negative PU (nnPU) learning algorithm and the density ratio-based PU (DRPU) approach.

\subsection{Loss Functions for nnPU}

The derivation of the training objective for the nnPU algorithm includes the choice of the loss function. In our experiments, we compare two popular loss functions: the sigmoid loss and the binary cross-entropy loss, also known as the logistic loss.

The sigmoid loss is defined as
 $$
 l_{sig}(t, y) = 1/(1 + exp(ty)),
 $$

while the logistic loss is given by
 $$
 l_{log}(t, y) = \ln(1 + exp(-ty)).
 $$

 Sigmoid loss is non-convex and satisfies the symmetry condition: $l(t, +1) + l(t, -1) = 1$. Indeed,

 $$
 l_{sig}(t, +1) + l_{sig}(t, -1) = \frac{1}{1 + e^t} + \frac{1}{1 + e^{-t}} = \frac{1 + e^{-t} + 1 + e^{t}}{(1 + e^t)(1 + e^{-t})} = \frac{1 + e^{-t} + 1 + e^{t}}{1 + e^{-t} + 1 + e^{t}} = 1.
 $$
 
The logistic loss is convex and satisfies the conditions required for unbiased PU risk estimation. Furthermore, its optimal prediction function can be derived by minimizing the conditional risk:

$$
R(f \mid x) = \log(1+e^{-f})\,p(Y=+1\mid x) + \log(1+e^{f})\,p(Y=-1\mid x).
$$

Differentiating with respect to $f$ yields:

$$
\frac{\partial}{\partial f} \Big[ \log(1+e^{-f})\,p(Y=+1\mid x) + \log(1+e^{f})\,p(Y=-1\mid x) \Big] =
-\frac{p(Y=+1\mid x)}{1+e^{f}} + \frac{e^{f}p(Y=-1\mid x)}{1+e^{f}}.
$$

Setting the derivative equal to zero gives:

$$
-p(Y=+1\mid x) + e^{f}p(Y=-1\mid x) = 0.
$$

Let $\eta(x)=p(Y=+1\mid x)$, and transform the above formula:

$$
e^f = \frac{p(Y=+1\mid x)}{p(Y=-1\mid x)} = \frac{\eta(x)}{1-\eta(x)},
$$

which provides the following formula for $f$:

$$
f(x) = \log\left( \frac{\eta(x)}{1-\eta(x)} \right).
$$

We can observe that this produces the Bayes classifier since:

$$
f(x) > 0 \Leftrightarrow \eta(x) > \frac{1}{2}.
$$

Therefore, both loss functions satisfy the conditions required for the unbiasedness of the PU risk estimator, as discussed in Section~\ref{sec:upu}. This means that these functions are valid choices for nnPU training.

\subsection{Prior Estimation Methods}

For the purpose of prior estimation $\hat{\pi}'$ from unlabeled data $X_U$, we consider and compare the following methods:

\begin{itemize}
    \item Kernel mean embedding based estimator KM2,
    \item Density ratio (DR)-based prior estimation with DR model trained on $X_P+X_U$ (positive and source unlabeled),
    \item Density ratio (DR)-based prior estimation with DR model trained on $X_P+X_{U'}$ (positive and target unlabeled),
    \item Maximum likelihood label shift (MLLS) using the nnPU model,
    \item Maximum likelihood label shift (MLLS) using the DRPU model.
\end{itemize}

\section{Methodology Overview}

We summarize all evaluated variants in Table~\ref{tab:methods-codes}. As baselines, we report the original implementations of \texttt{nnPU} and \texttt{DRPU}. The nnPU method does not provide any adaptation to label shift. Additionally, we include a variant of nnPU in which the source prior is estimated using KM2 instead of being assumed known (\texttt{nnPU+KM2}). This is the only method where we relax our general assumption of known source prior. The purpose of including this variant is to verify whether this assumption can considerably influence performance in favor of methods making use of this information.

Next, we evaluate the threshold adjustment strategy proposed in Section~\ref{sec:ta}, which modifies the decision rule at test time without retraining the underlying model. These variants are denoted by the \texttt{TA} keyword and depend on the method used for prior estimation. The \texttt{True} variant refers to the true empirical class prior in the target data and is included solely for comparison with estimator-based variants, since in realistic scenarios only estimated priors are available. We therefore consider two practical estimators: the widely used \texttt{KM2} estimator and the density ratio-based estimator associated with the DRPU methodology (\texttt{DR}). We utilize the same models as in nnPU and DRPU, however for nnPU, threshold adjustment replaces the default classification threshold of $0.5$. Since DRPU already incorporates a threshold adaptation mechanism, the baseline \texttt{DRPU} itself belongs to this group. We additionally evaluate variants where the original density ratio estimate is replaced the KM2 estimate (\texttt{DRPU+TA+KM2}). We classify all these methods as one group, related to threshold adaptation.

We also combine both learning frameworks with the EM-based MLLS procedure described in Section~\ref{posterior-adjust} (\texttt{nnPU+MLLS} and \texttt{DRPU+MLLS}). In these variants, the underlying models are trained exactly as in standard nnPU and DRPU. Posterior probabilities are computed in the usual way but subsequently adjusted using the MLLS procedure before generating final predictions. All posterior scores are thresholded at $0.5$. The application of MLLS to PU learning under label shift is proposed in this work, and these two methods constitute the group of posterior adjustment approaches.

Finally, to assess the mixed-training strategy proposed in Section~\ref{target-retrain}, we include one retraining variant for each learning objective (\texttt{nnPU+Target} and \texttt{DRPU+Target}). In these methods, the model is trained using the positive labeled set $X_P$ together with the unlabeled set $X_{U'}$ sampled from the target distribution, instead of the source unlabeled set $X_U$. In our setting, we assume knowledge of the source prior but not the target prior. Here, before training, we need to additionally estimate the target prior, as it is required for nnPU training and can be also utilized for DRPU training. For both models, the target prior is estimated using the KM2 method. Apart from replacing the training dataset, the standard nnPU and DRPU training procedures remain unchanged. The nnPU model, which does not inherently address label shift, receives no further adaptation. For DRPU, training is performed on $X_P + X_{U'}$ with parameter $\alpha$ set to the KM2-estimated target prior. Although standard DRPU includes a threshold adaptation step, this step is omitted here because the model is trained directly on target data, so label shift adaptation is not needed. Consequently, a fixed threshold of $0.5$ is used. At test time, predictions rely on the density ratio estimate of the target prior, as in standard DRPU. The KM2 estimate must be used during training as it is the only estimate available prior to model fitting. These two methods belong to target retraining group.

Additionally, each method based on the nnPU learning framework is evaluated using two loss functions: sigmoid and logistic. Note that the baseline methods \texttt{nnPU} and \texttt{nnPU+KM2} do not belong to any of the outlined methodology groups in this work, as they do not implement any label adaptation mechanism, nevertheless, they are included for reference and comparison. 

\begin{table}[htbp]
  \centering
  \small
  \setlength{\tabcolsep}{6pt}
  \renewcommand{\arraystretch}{0.85} % reduce vertical spacing
  \begin{tabular}{
      p{2.7cm}|
      p{1cm}|
      p{11cm}
  }
    \toprule
    \textbf{Method} & \textbf{Group} & \textbf{Explanation} \\
    \midrule
    \texttt{nnPU} & - &
    Baseline non-negative PU classifier trained on $X_P + X_U$, this method does not contain any label shift adaptation mechanism, the source prior is known \\
    \midrule
    \texttt{nnPU+KM2} & - &
    Baseline non-negative PU classifier trained on $X_P + X_U$, this method does not contain any label shift adaptation mechanism, the source prior is estimated with KM2 \\
    \midrule
    \texttt{nnPU+TA+KM2} & 1 &
    nnPU with threshold adjustment using target prior estimated by KM2 \\
    \midrule
    \texttt{nnPU+TA+DRE} &  1 &
    nnPU with threshold adjustment using target prior estimated by DR-based estimator \\
    \midrule
    \texttt{nnPU+MLLS} & 2 &
    nnPU combined with EM-based MLLS label shift adaptation, the adjusted posteriors are subjected to $0.5$ threshold \\
    \midrule
    \texttt{nnPU+Target} & 3 &
    nnPU procedure applied to $X_P + X_{U'}$, class prior used for training is estimated with KM2 \\
    \midrule
    \texttt{DRPU} & 1 &
    Baseline density ratio PU method trained on $X_P + X_U$, it includes threshold adaptation with target prior estimated by DR-based estimator \\
    \midrule
    \texttt{DRPU+TA+KM2} & 1 &
    DRPU with threshold adjustment using target prior estimated by KM2 instead of the DR estimate \\
    \midrule
    \texttt{DRPU+MLLS} & 2 &
    DRPU combined with EM-based MLLS label shift adaptation, the adjusted posteriors are subjected to $0.5$ threshold \\
    \midrule
    \texttt{DRPU+Target} & 3 &
    DRPU procedure on $X_P + X_{U'}$, without the threshold adjustment step, the class prior used during training is estimated with KM2, whereas at test time it is estimated using the DR-based method \\
    \bottomrule
  \end{tabular}
  \caption{Methods and codes used in experiments. The keyword \texttt{Target} means that the model is retrained on the mixed set $X_P + X_{U'}$, while the rest is trained on the source PU data $X_P + X_U$.} The \textit{Group} column indicates the method category: 1 — threshold adjustment, 2 — posterior adjustment, 3 — target retraining.
  \label{tab:methods-codes}
\end{table}

\newpage
\section{Evaluation of Results}

In this section, we describe the metrics used to evaluate the performance of the proposed approach. We consider both the estimation error of class prior and the effectiveness of the PU classifier.

\subsection{Evaluation of Prior Estimation}

To assess the estimation error of class prior $\hat{\pi}$, we repeat the experiment $K$ times and compute the mean absolute error (MAE):

\begin{equation}
MAE = \frac{1}{K} \sum_{i=1}^{K} \left| \pi - \hat{\pi}_{i} \right|.
\end{equation}

The variability of the estimation process is quantified using the standard error of MAE:

\begin{equation}
SE = \frac{s}{\sqrt{K}},
\end{equation}

where $s^2$ is a sample variance of the absolute errors:

$$
s^2 = \frac{1}{K - 1} \sum_{i=1}^{K} \left( \left| \pi - \hat{\pi}_{i} \right| - MAE \right)^2.
$$


\subsection{Evaluation of Classification Performance}

In addition to prior estimation, the quality of the classifier is evaluated in the binary classification setting. The accuracy is calculated as the proportion of correctly classified samples:

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN},
$$

where $TP$, $TN$, $FP$, and $FN$ denote the counts of true positives, true negatives, false positives, and false negatives, respectively.


Since label shift alters class proportions in the target data, the resulting datasets may become highly imbalanced. In such cases, accuracy alone may not adequately reflect classifier performance. Therefore, in selected experiments we also report the balanced accuracy, defined as:

$$
Balanced\ Accuracy = \frac{TPR + TNR}{2},
$$

where $TPR$ (True Positive Rate) is defined as: 

$$
TPR = \frac{TP}{TP + FN},
$$

and $TNR$ (True Negative Rate) is defined as:

$$
TNR = \frac{TN}{TN + FP}.
$$

Balanced accuracy gives equal importance to both classes. However, it is important to stress that the considered PU learning objectives are constructed to optimize the standard classification risk, which is closely related to overall accuracy. For this reason, accuracy remains the primary evaluation metric in our analysis, while balanced accuracy serves as a complementary measure.


Additionally, precision quantifies the proportion of positive predictions that are correct and is defined as:

$$
Precision = \frac{TP}{TP + FP}.
$$

Recall, also known as sensitivity is equal to True Positive Rate (TPR), it measures the proportion of actual positive samples that are correctly identified. F1-score combines Precision and Recall into a single metric by calculating their harmonic mean:

$$
F1-score = 2 \times \frac{Precision \times Recall}{Precision + Recall}.
$$

\subsection{Receiver Operating Characteristic Analysis}

In our experiments, decision rules are modified by adjusting classification thresholds in order to account for label shift. To further analyze and visualize the impact of threshold selection on classifier behavior, we examine the Receiver Operating Characteristic (ROC) curve. The ROC curve illustrates the trade-off between the True Positive Rate (TPR) and the False Positive Rate (FPR) across varying decision thresholds. Each point on the ROC curve corresponds to a specific threshold applied to the classifier's output scores.

A commonly used criterion for selecting an optimal decision boundary is the maximum of the Youden statistic \citep{youden1950}. The Youden index is defined as:

$$
J = TPR - FPR.
$$

Maximizing $J$ corresponds to maximizing the vertical distance between the ROC curve and the diagonal line representing random classification. It can be shown that maximizing the Youden statistic is equivalent to maximizing the balanced accuracy, since

$$
Balanced\ Accuracy = \frac{TPR + TNR}{2}
= \frac{TPR + (1 - FPR)}{2}
= \frac{1 + (TPR - FPR)}{2}.
$$

Thus, maximizing $J = TPR - FPR$ directly maximizes balanced accuracy. For a given model, this allows us to find the optimal balanced accuracy threshold. This threshold can only be identified when the true target labels are available, as the computation of $TPR$ and $FPR$ requires labeled data. Therefore, in our analysis, it serves as a benchmark against which we compare the thresholds selected by our label shift adaptation procedures to assess how closely the proposed methods approximate the optimal balanced accuracy maximization solution. We emphasize that our threshold adjustment methods aim to improve overall accuracy rather than balanced accuracy. Nevertheless, we find such an analysis insightful, especially for high imbalanced target data.

\chapter{Datasets} \label{chap:datasets}

In this chapter, we present the datasets used in the experiments, including both synthetic and real-world data. To conform to the binary classification setting outlined in our problem statement, some datasets were binarized. All datasets include labels for every observation, however, in each experiment, we generate positive and unlabeled sets from the available training data. As a result, some training labels are intentionally removed.

Furthermore, in most experiments, only a subset of each dataset is used. This allows for simulating label shift by altering class proportions in the training and test sets. The procedure for adapting the datasets to the specific experimental settings is described in detail in Chapter~\ref{chap:exp_setting}.

\section{Gaussian Dataset}

The first dataset is a synthetic Gaussian dataset. It consists of 10-dimensional feature vectors sampled from two multivariate normal distributions:

$$
p_p \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{10}), \quad 
p_n \sim \mathcal{N}(0.8 \cdot \mathbf{1}, \mathbf{I}_{10}),
$$

where $p_p$ and $p_n$ represent the positive and negative class distributions, respectively.  
In the PU setting, the labeled positive samples $X_P$ are drawn from $p_p(x)$, while the unlabeled samples $X_U$ come from the overall mixture distribution $p(x) = \pi p_p(x) + (1 - \pi) p_n(x)$. This setup allows generating datasets of arbitrary size and can be reused with different training and test class priors, enabling controlled experiments across a grid of parameter configurations.

\section{MNIST Dataset}

The second dataset used in our experiments is the MNIST dataset ((Modified National Institute of Standards and Technology database) \citep{lecun2010}, which consists of grayscale images of handwritten digits.  
It contains \(60,000\) training and \(10,000\) test samples, with each image represented as a \(28 \times 28\) pixel grid.  
To fit the binary classification setting, the digits were divided into two classes: even digits \((0, 2, 4, 6, 8)\) form the positive class, while odd digits \((1, 3, 5, 7, 9)\) form the negative class.  
In the PU setting, the positive samples \(X_P\) are randomly drawn from the subset of even-digit images, whereas the unlabeled set \(X_U\) is sampled from the full distribution of digits, containing both positive and negative examples.

An example subset of images from MNIST is shown in Figure~\ref{fig:mnist}.

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.4\textwidth]{2. thesis/img/mnist.png}
    \caption{Sample images from the MNIST dataset.}
    \label{fig:mnist}
\end{figure}

\section{FashionMNIST Dataset}

Similarly, to MNIST dataset, FashionMNIST \citep{xiao2017} is the set of \(28 \times 28\) grayscale image, associated with a label from 10 classes, but instead of numbers, the images represent Zalando's articles, including the following categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot. The transformation to binary setting was done by combining the upper parts of clothing into the positive class (T-shirt/top, Pullover, Dress, Coat, Shirt), and treat the lower parts as the negative class. The training set contains \(60,000\) samples and test \(10,000\) samples. 

An example subset of images from FashionMNIST is shown in Figure~\ref{fig:fashionmnist}.

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.7\textwidth]{2. thesis/img/fashionmnist_ex.png}
    \caption{Sample images from the FashionMNIST dataset.}
    \label{fig:fashionmnist}
\end{figure}

\section{ChestXRay Dataset} 

The ChestXRay dataset, introduced by ~\cite{Kermany2018MedicalDL}, addresses the task of pneumonia detection from pediatric chest X-ray images. Pneumonia is one of the leading causes of childhood mortality worldwide, which makes accurate and early diagnosis particularly important. The dataset consists of chest X-ray images collected from children and includes cases of both bacterial and viral pneumonia, as well as normal (healthy) images. All diagnoses were initially labeled by two expert physicians and later verified by a third independent expert. 

All images were resized to a resolution of $640 \times 640$ pixels and automatically oriented. In our experiments, the examples of pneumonia are treated as the positive class, while normal chest X-ray images serve as the negative class. The training set contains \(4,077\) samples and test set consists of \(582\) samples.

The example observations from the dataset are presented in Figure~\ref{fig:chestxray_samples}

\begin{figure}[h]
\hfill
\hspace*{-2cm}  
\subfigure[Pneumonia]{\includegraphics[width=6cm]{2. thesis/img/chestxray_pneumonia.png}}
\hfill
\subfigure[Normal]{\includegraphics[width=6cm]{2. thesis/img/chestxray_normal.png}}
\hfill
\caption{Sample images from the ChestXRay dataset.}
\label{fig:chestxray_samples}
\end{figure}


\section{Electricity Dataset}

The Electricity dataset \cite{harries1999splice} was collected from the Australian New South Wales Electricity Market, where prices are dynamic and influenced by supply and demand of the market, updated every five minutes. The dataset, originally referred to as \texttt{ELEC2}, contains 38,474 instances dated from 7 May 1996 to 5 December 1998.  

Each instance corresponds to a 30-minute interval, yielding 48 instances per day. The dataset includes the following features: day of the week, timestamp, New South Wales electricity demand, Victoria electricity demand, scheduled electricity transfer between states, and the class label. The class label indicates whether the New South Wales price moves up or down relative to a moving average over the past 24 hours.

% \section{Covertype Dataset}

% The Covertype dataset concerns predicting forest cover type using only cartographic variables, without remote sensing data. The actual forest cover for each $30 \times 30$ meter cell was determined using the US Forest Service (USFS) Region 2 Resource Information System (RIS). Independent variables were derived from US Geological Survey (USGS) and USFS data. The dataset contains both continuous variables and binary indicators for qualitative variables such as wilderness areas and soil types.  

% The study area includes four wilderness regions within the Roosevelt National Forest in northern Colorado, where forest cover is largely determined by ecological factors rather than human management. The original dataset includes several forest cover types, such as Spruce/Fir, Lodgepole Pine, Ponderosa Pine, Cottonwood/Willow, Aspen, Douglas-fir, and Krummholz.  

% For our experiments, we use the transformed version of the dataset prepared by \cite{grinsztajn2022}, where the target was binarized.

\section{CIFAR-10 Dataset}

The CIFAR-10 \cite{Krizhevsky09} dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The images present one of the 10 possible objects: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
We transform the dataset to binary classification problem by composing a positive class out of animals (birds, cats, deers, dogs, frogs, horses), while the negative class is formed by remaining observations representing vehicles (airplanes, automobiles, ships, trucks).

An example subset of images from CIFAR-10 is shown in Figure~\ref{fig:cifar10}.

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.7\textwidth]{2. thesis/img/cifar10_ex.png}
    \caption{Sample images from the CIFAR-10 dataset.}
    \label{fig:cifar10}
\end{figure}

\section{SMS Spam Dataset}

The SMS Spam Collection \cite{almeida2011} is a publicly available dataset of 5,574 English SMS messages, labeled as either legitimate or spam. The messages are real-world texts, and they are embedded before training.

\newpage

\section{Datasets Overview}

Table \ref{tab:datasets_stats} presents the dataset statistics: the total number of samples $n$, the number of positive observations $n_p$, and the empirical prior calculated as $\hat{\pi}=\frac{n_p}{n}$ for both the training and test splits (in our setting referred to as source and target). The statistics are reported after transforming the original datasets into binary classification problems according to the rules described in the previous sections.

% Requires: \usepackage{booktabs}, \usepackage{multirow}
\begin{table}[htbp]
  \centering
  \small
  \setlength{\tabcolsep}{6pt}
  \begin{tabular}{
      p{2.2cm}   % Dataset
      p{1.5cm}   % Type
      p{1.9cm}   % Train: #samples
      p{1.9cm}   % Train: #pos
      p{1.2cm}   % Train: Prior
      p{1.9cm}   % Test: #samples
      p{1.9cm}   % Test: #pos
      p{1.2cm}   % Test: Prior
  }
    \toprule
    \multirow{2}{*}{\textbf{Dataset}} &
    \multirow{2}{*}{\textbf{Type}} &
    \multicolumn{3}{c}{\textbf{Source}} &
    \multicolumn{3}{c}{\textbf{Target}} \\
    \cmidrule(lr){3-5}\cmidrule(lr){6-8}
    & & \textbf{\# samples} & \textbf{\# positive} & \textbf{Prior} & \textbf{\# samples} & \textbf{\# positive} & \textbf{Prior} \\
    \midrule
    \texttt{Gauss} & Tabular & 5,000 & 2,500 & 0.5 & 5,000 & 2,500 & 0.5 \\
    \texttt{MNIST} & Image & 60,000 & 30,000 & 0.5 & 10,000 & 5,000 & 0.5 \\
    \texttt{FashionMNIST} & Image & 60,000 & 30,000 & 0.5 & 10,000 & 5,000 & 0.5 \\
    \texttt{ChestXRay} & Image & 4,077 & 2,973 & 0.729 & 582 & 411 & 0.706 \\
    \texttt{Electricity} & Tabular & 30,779 & 15,372 & 0.5 & 7,695 & 3,865 & 0.5 \\
    % \texttt{Covertype} & 453,281 & 226,748 & 0.5 & 113,321 & 56,553 & 0.5 \\
    \texttt{CIFAR10} & Image & 50,000 & 30,000 & 0.6 & 10,000 & 6,000 & 0.6 \\
    \texttt{SMSSpam} & Text & 4,459 & 602 & 0.135 & 1,115 & 145 & 0.130 \\
    
    % Add more rows below as needed
    % DatasetName & total & train_n & train_pos & train_prior & test_n & test_pos & test_prior \\
    \bottomrule
  \end{tabular}
  \caption{Selected datasets statistics for binary classification setting.}
  \label{tab:datasets_stats} 
\end{table}

We transformed all observations into vector embeddings before applying the PU learning methods. The embeddings were reused from the implementation of \cite{mielniczuk2024}. For the text dataset (\texttt{SMSSpam}), sentence representations were generated using the \texttt{all-MiniLM-L6-v2} model, which is the fine-tuned version of MiniLM language model \citep{wang2020}. For all image datasets (\texttt{MNIST}, \texttt{FashionMNIST}, \texttt{ChestXRay}, and \texttt{CIFAR10}), images were processed using the SwiftFormer (version \texttt{swiftformer-xs}) model \citep{shaker2023}, and the resulting feature representation was obtained by flattening the final hidden-layer activations into a fixed-length embedding vector. For the tabular dataset, \texttt{Electricity}, the original numerical attributes were only standardized and used directly as model inputs.
 

\chapter{Experiments Setting} \label{chap:exp_setting}

In this Chapter, we describe in detail the conducted experiments, including the procedure of creating datasets to match specific label shift scenarios, the architecture of models, and the list of all experiments with tested configurations.

\section{Simulating Label Shift}

We extend the case-control scenario described in \cite{mielniczuk2024} by introducing the ability to control the class proportions in the PU datasets, allowing the simulation of label shift in the data. We define an algorithm that shifts the data to target priors by randomly removing certain observations from the full dataset.

Let us assume that we have a PU dataset containing $\Tilde{n}$ samples, including $\Tilde{n}_p$ positively labeled samples and $\Tilde{n}_u$ unlabeled samples. Among the unlabeled samples, there are $\Tilde{n}_{up}$ positive samples and $\Tilde{n}_{un}$ negative samples, so that $\Tilde{n} = \Tilde{n}_p + \Tilde{n}_u = \Tilde{n}_p + \Tilde{n}_{up} + \Tilde{n}_{un}$.  Let $\Tilde{\pi}$ denote the class prior in the underlying data distribution.

In a given experiment, our aim is to construct a dataset in which we control the expected fraction of labeled examples among all positive observations, denoted by $c$ and referred to as the label frequency. At the same time, we seek to achieve a class prior different from the original one, denoted by $\pi$, in order to simulate label shift. Additionally, in some experiments, we fix the total number of observations to $n = n_p + n_{up} + n_{un}$.

Under the case-control sampling scheme, the label frequency is defined as:

$$
c = \frac{\mathbb{E}[n_p]}{\mathbb{E}[n_p] + \mathbb{E}[n_{up}]}.
$$ 

Furthermore, the class prior in the unlabeled set satisfies:
$$
\pi = \frac{\mathbb{E}[n_{up}]}{\mathbb{E}[n_{up}] + \mathbb{E}[n_{un}]},
$$
which reflects the proportion of positive observations among unlabeled samples.

The expected numbers of unlabeled positive and negative observations are given by:
$$
\mathbb{E}[n_{up}] = \pi \, \mathbb{E}[n_u], 
\qquad
\mathbb{E}[n_{un}] = (1-\pi)\,\mathbb{E}[n_u],
$$
where $n_u = n_{up} + n_{un}$ denotes the size of the unlabeled set.

Therefore, it is necessary to compute the parameters $n^{(new)}_p = \mathbb{E}[n_p]$, $n^{(new)}_{up} = \mathbb{E}[n_{up}]$, and $n^{(new)}_{un} = \mathbb{E}[n_{un}]$. These quantities define how many observations are sampled from the corresponding subsets of the full dataset. We sample $n^{(new)}_p + n^{(new)}_{up}$ positive observations and $n^{(new)}_{un}$ negative observations. These counts can be obtained by solving the following system of equations:

$$
\left\{
\begin{aligned}
c &= \frac{n^{(new)}_p}{n^{(new)}_p + n^{(new)}_{up}} && \text{\quad\quad (label frequency)} \\
n^{(new)} &= n^{(new)}_p + n^{(new)}_{up} + n^{(new)}_{un} && \text{\quad\quad (number of samples)} \\
\pi^{(new)} &= \frac{n^{(new)}_{up}}{n^{(new)}_{up} + n^{(new)}_{un}} && \text{\quad\quad (shifted prior)}
\end{aligned}
\right.
$$

The solution to this problem can be formulated as the algorithm presented in Algorithm~\ref{alg:shift_data}, which calculates the required sample sizes for each class to achieve a specified label frequency, total number of samples, and shifted prior.

\begin{algorithm}[htbp]
\caption{Shift Data Algorithm (Case-Control Scenario)}\label{alg:shift_data}
\begin{algorithmic}
\Require Label frequency $c$, original prior $\pi$, total number of samples $n$, shifted prior $\pi^{(new)}$, desired number of samples $n^{(new)}$
\Ensure Counts $n^{(new)}_p$, $n^{(new)}_{up}$, $n^{(new)}_{un}$

\State $A \gets \left(1 - c + c \cdot \pi^{(new)} \right)^{-1}$ 
\Comment{\parbox[t]{10cm}{\textit{scaling factor}}}

\State $n^{(new)}_p \gets A \cdot c \cdot \pi^{(new)} \cdot n^{(new)}$ 
\Comment{\parbox[t]{10cm}{\textit{number of labeled positives: $c \cdot \pi^{(new)} \cdot n^{(new)}$ scaled by $A$}}}

\State $n^{(new)}_u \gets A \cdot (1 - c) \cdot n^{(new)}$ 
\Comment{\parbox[t]{10cm}{\textit{number of unlabeled samples}}}

\State $n^{(new)}_{un} \gets (1 - \pi^{(new)})n^{(new)}_{u}$
\State $n^{(new)}_{up} \gets \pi^{(new)} n^{(new)}_u$

% \Statex
% \Statex \textbf{Assertions:}
% \Statex \quad $n'_p + n'_{up} \leq n_p + n_{up}$ \quad \textit{(do not exceed available positive samples)}
% \Statex \quad $n'_{un} \leq n_{un}$ \quad \textit{(do not exceed available negative samples)}

\end{algorithmic}
\end{algorithm}

\begin{example}
To illustrate the application of the algorithm, consider the case where we wish to simulate a dataset with a prior probability of positive examples $\pi' = 0.2$, a label frequency $c = 0.5$, and a total number of samples $n' = 1500$.

First, we compute the scaling factor:
\[
A = \left(1 - c + c \cdot \pi^{(new)} \right)^{-1} = \left(1 - 0.5 + 0.5 \cdot 0.2 \right)^{-1} = (0.6)^{-1} = \frac{5}{3}.
\]

Then, the number of labeled positive samples is:
\[
n^{(new)} _{p} = A \cdot c \cdot \pi^{(new)} \cdot n^{(new)} = \frac{5}{3} \cdot \frac{1}{2} \cdot \frac{1}{5} \cdot 1500 = 250.
\]

The number of unlabeled samples is:
\[
n^{(new)}_{u} = A \cdot (1 - c) \cdot n^{(new)} = \frac{5}{3} \cdot \frac{1}{2} \cdot 1500 = 1250.
\]

Among these unlabeled samples, the expected number of negatives is:
\[
n^{(new)}_{un} = (1 - \pi^{(new)}) \cdot n^{(new)}_{u} = 0.8 \cdot 1250 = 1000,
\]
and the expected number of positives is:
\[
n^{(new)}_{up} = \pi^{(new)} \cdot n^{(new)}_{u} = 0.2 \cdot 1250 = 250.
\]

Finally, we verify the prior probability:
\[
\pi^{(new)} = \frac{n^{(new)}_{up}}{n^{(new)}_{u}} = \frac{250}{1250} = 0.2,
\]
and the label frequency:
\[
c = \frac{n^{(new)}_{p}}{n^{(new)}_{p} + n^{(new)}_{up}} = \frac{250}{250 + 250} = 0.5.
\]
\end{example}


\section{Model}

In a single experiment, there are 2 models trained: one with the objective function for nnPU risk (Formula \ref{nnpu}) and the second with the objective function of DRPU risk (Formula \ref{drpu}). In DRPU risk the following Bregman generator function $f$ was used:

$$
f(x) = \frac{(x-1)^2}{2}.
$$

The derivative and dual form of that function is the following:

\begin{gather*}
f'(x) = x - 1, \\
f^*(x) = xf'(x) - f(x) = \frac{x^2-1}{2}.
\end{gather*}

For both models the same architecture of Multi-Layer Perceptron (MLP), following the design proposed by \cite{kiryo2017}, is used. The model takes as input flattened feature vectors of dimension $d$ and maps them through a series of transformations to a single scalar output. The network consists of five linear layers, where the first four hidden layers have 300 units each and are followed by Batch Normalization and the ReLU activation function $f(x) = \max(0, x)$ \citep{agarap2018}. To avoid redundancy, all linear layers before the final output layer are defined without bias terms, since Batch Normalization layers include learnable parameters. The final layer is a single-unit linear layer. 

The objectives were minimized using the Adam stochastic optimization algorithm \citep{kingma2014} with learning rate parameter $\alpha=10e^{-5}$, decay rates $\beta_1=0.9, \beta_2=0.999$, and weight decay $\lambda = 0.005$. All models were trained for 50 epochs.

\section{Experiments Configuration}

Table~\ref{tab:controlled_experiments} presents the parameter configurations used in the controlled experimental setting. For each dataset, experiments were performed over the Cartesian product of all possible parameter values. Algorithm~\ref{alg:shift_data} was utilized to generate the corresponding experimental data according to the selected configurations.

\begin{table}[htbp]
  \centering
  \small
  \setlength{\tabcolsep}{6pt}
  \begin{tabular}{
      p{2.5cm}   % Dataset
      p{1.9cm}   % Train: #samples
      p{2.8cm}   % Train: Prior
      p{2.8cm}   % Train: Label frequency
      p{1.9cm}   % Test: #samples
      p{2.8cm}   % Test: Prior
  }
    \toprule
    \multirow{2}{*}{\textbf{Dataset}} &
    \multicolumn{3}{c}{\textbf{Source ($X_P+X_{U}$)}} &
    \multicolumn{2}{c}{\textbf{Target ($X_{U'}$)}} \\
    \cmidrule(lr){2-4}\cmidrule(lr){5-6}
    & \textbf{\# samples} & \textbf{Prior} & \textbf{Label freq.} & \textbf{\# samples} & \textbf{Prior} \\
    \midrule
    \texttt{Gauss} & 5,000 & 0.2, 0.4, 0.6, 0.8 & 0.5 & 5,000 & 0.2, 0.4, 0.6, 0.8 \\
    \texttt{MNIST} & 5,000 & 0.5 & 0.5 & 5,000 & 0.2, 0.4, 0.6, 0.8 \\
    \texttt{FashionMNIST} & 5,000 & 0.5 & 0.5 & 5,000 & 0.2, 0.4, 0.6, 0.8 \\
    \texttt{ChestXRay} & 5,000 & 0.5 & 0.5 & 5,000 & 0.2, 0.4, 0.6, 0.8 \\
    \texttt{CIFAR-10} & 5,000 & 0.5 & 0.5 & 5,000 & 0.2, 0.4, 0.6, 0.8 \\
    \texttt{Electricity} & 5,000 & 0.5 & 0.5 & 5,000 & 0.2, 0.4, 0.6, 0.8 \\
    \texttt{SMSSPam} & 5,000 & 0.5 & 0.5 & 5,000 & 0.2, 0.4, 0.6, 0.8 \\
    \bottomrule
  \end{tabular}
  \caption{Configurations for experiments.}
  \label{tab:controlled_experiments}
\end{table}

It is important to note that the conducted experiments were computationally very challenging. The main computational bottleneck was the optimization procedure used in the KM2 algorithm, whose runtime increased substantially for larger datasets. Due to limited computational resources, it was necessary to restrict the scope of the experiments. On a standard local machine, the synthetic data experiment \texttt{Gauss} required approximately 20 hours to complete, while the real data experiments required a similar amount of time.

As a consequence, we limited the synthetic experiments to a single synthetic dataset and reduced the number of configurations evaluated on real datasets. At the same time, we aimed to preserve statistical reliability by maintaining at least 10 repetitions for each experiment. For computational reasons, we also did not vary the label frequency parameter and fixed it to the default value of $0.5$ in all configurations. Additionally, for all real datasets, we sampled only 5,000 observations in order to further reduce computational costs.

The selected configurations and the label shift simulation algorithm aim to provide the fairest possible comparison setting. However, it is important to stress that an ideal experimental setup is impossible to achieve. Although all configurations contain the same total number of samples, various combinations of source and target priors result in different numbers of labeled positive observations, unlabeled positive observations, and unlabeled negative observations. Naturally, learning becomes more challenging when the labeled positive set is substantially smaller than the unlabeled set. On the other hand, classification may become easier when the unlabeled set is dominated by a single class, particularly when it contains mostly negative samples. When interpreting the experimental results, it should therefore be kept in mind that some prior configurations may be more challenging than others. In particular, extreme prior values may lead to highly imbalanced training data and consequently make the learning task more difficult. 

Figure~\ref{fig:n_of_samples_ex} illustrates how the sizes of the individual subsets vary as a function of the class prior. The three curves represent the number of labeled positive observations $n_p$, the total number of unlabeled observations $n_u$, and the number of unlabeled negative observations $n_{un}$. Since the total dataset size is fixed, increasing the class prior results in a larger labeled positive set and both the unlabeled set and the number of negative observations decrease.

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.9 \textwidth]{2. thesis/img/n_of_samples_ex.png}
    \caption{Sizes of the labeled positive set $n_p$, the unlabeled set $n_u$, and the unlabeled negative subset $n_{un}$ as a function of the class prior. The total number of observations is fixed at $n=5,000$.}
    \label{fig:n_of_samples_ex}
\end{figure}

Additionally, for the target retraining method, we construct an alternative training dataset whose size depends on the specific experimental configuration and is therefore not fixed. Table~\ref{tab:ex_no_obs} presents the numbers of observations in the individual datasets used in the experiments when the source prior is equal to $0.2$ and the target prior is shifted to $0.4$. Throughout the experiments, we construct three experimental datasets: $X_P+X_U$, used for training the majority of methods, $X_P+X_{U'}$, used for target retraining methods (here the total number of observations used for training exceeds 5,000, as explained in Section~\ref{target-retrain}), and $X_{U'}$, which serves as the test dataset for all methods.

\begin{table}[htbp]
  \centering
  \small
  \setlength{\tabcolsep}{6pt}
  \begin{tabular}{
      p{2cm}
      p{3.5cm}
      p{3.5cm}
      p{3.5cm}
  }
    \toprule
    \textbf{Set} & \textbf{\# samples} & \textbf{\# labeled pos.} & \textbf{\# unlabeled pos.} \\
    \midrule
    $X_P+X_U$        & 5000 & 834 & 834 \\
    $X_P+X_{U'}$     & 5834 & 834 & 2000 \\
    $X_{U'}$         & 5000 & 0 & 2000 \\
    \bottomrule
  \end{tabular}
  \caption{Numbers of observations in the datasets used in a sample experimental configuration ($\pi=0.2$, $\pi'=0.4$, $n=n'=5000$, $c=0.5$).}
  \label{tab:ex_no_obs}
\end{table}



% \subsection{Experiments on Entire Datasets}

% Beside the controlled experiments, we also conduct experiments using the entire training and test sets, without specifying the number of samples or the desired training or test prior. In this setting, the only adjustable parameter is the label frequency, for which we typically consider the values $\{0.1, 0.25, 0.5, 0.75, 0.9\}$. This type of experiment is performed exclusively on the non-synthetic datasets.



\chapter{Results for Synthetic Data}

In this chapter, we present the results of the experiments described in the previous chapters conducted on a synthetic Gaussian dataset. The analysis is organized into several categories. First, we examine the impact of the choice of loss function on the performance of the nnPU model. Next, we evaluate the quality of class prior estimation and analyze its influence on the training process and the resulting classification performance. Furthermore, we compare the classification performance within each of the three methodology groups outlined in Chapter~\ref{chap:methodology} and identify the best-performing methods.

Additionally, we study the distributions of model scores produced by the nnPU and DRPU models. We also present an analysis of the Receiver Operating Characteristic (ROC) curve for a selected experiment. Finally, we deepen the analysis of threshold choices in threshold adaptation methods by evaluating accuracies as a function of the threshold and analyzing the distribution of model errors.

\section{Impact of Loss Function Choice on nnPU Performance}

First, we compare the results of nnPU models using two different loss functions: sigmoid loss and binary cross-entropy loss. We calculated the average accuracies over ten repetitions of the same experiment, each with a different random seed. Figure~\ref{fig:gauss_accuracy_loss_ta_05} presents the results achieved by the classical nnPU model without any label shift adaptation, the classical DRPU methodology with built-in adaptation to label shift using threshold adjustment with the target prior estimated by the DR estimator, and other methods discussed in Section~\ref{sec:ta} that rely on threshold adaptation. Figure~\ref{fig:gauss_accuracy_loss_target_05} focuses on methods from the remaining two groups: MLLS, which adjusts posterior scores, and the target retraining method.

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_accuracy_loss_ta.png}
%     \caption{Comparison of sigmoid (Sig) and binary cross-entropy (CE) losses for threshold adjustment methods on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
%     \label{fig:gauss_accuracy_loss_ta_05}
% \end{figure}

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_accuracy_loss_target.png}
%     \caption{Comparison of sigmoid (Sig) and binary cross-entropy (CE) losses for MLLS and target retraining methods on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
%     \label{fig:gauss_accuracy_loss_target_05}
% \end{figure}

We observe unstable behavior of the MLLS method, which is based on the Expectation-Maximization algorithm. In some cases, the estimates do not converge to reasonable values, instead, the estimated prior converges to either 0 or 1. As a result, the model classifies all observations into a single class. To address this, we filtered out such results from our analysis. More strictly, we included an MLLS result only if the estimated prior satisfies the following condition: $|\hat{\pi}' - \pi'| < 0.15$, ensuring that the estimated prior is a valid estimate. For MLLS, we aggregate the results in the same manner as for the other methods, but only over the subset of validated experiments. In the plots, we annotate the number of experiments where MLLS converged correctly next to the graph points. For example, $5/10$ indicates that out of ten repetitions, the MLLS procedure converged according to our condition five times. For some configurations, there were no valid MLLS results, which is why some configurations miss the MLLS results at all. We discuss potential causes of this behavior in Section~\ref{sec:res-2-3}, which is devoted to posterior adjustment methods.

Comparing the results of individual methods across the two loss functions, we observe that, in almost all cases, better performance is achieved with the sigmoid loss. Single configurations exist where binary cross-entropy outperforms sigmoid loss. Generally, when the source prior is set to 0.8, the performance of sigmoid loss decreases and binary cross-entropy performs better. For other target prior values, sigmoid loss outperforms binary cross-entropy. Therefore, for simplification, we use nnPU with sigmoid loss in all further analyses.

\section{Influence of Prior Estimation on Classification Performance}

In this section, we analyze the group of methods based on threshold adaptation. These methods require an additional step consisting of estimating the target class prior. Figure~\ref{fig:gauss_mae_05} presents the mean absolute errors of target prior estimates obtained using different estimation methods. Similarly to the previous section, MLLS estimates are subjected to a filtering criterion: $|\hat{\pi}' - \pi'| < 0.15$.

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_mae.png}
%     \caption{Mean absolute error of prior estimation on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
%     \label{fig:gauss_mae_05}
% \end{figure}

Density ratio estimates are computed using two variants of the ratio estimator: one trained on $X_P + X_U$ (\texttt{DRE}) and another trained on $X_P + X_{U'}$ (\texttt{DRE (Target)}). In general, relatively accurate prior estimates can be achieved, with the estimation error not exceeding approximately 0.08. Density ratio based estimation performs very well overall, both when training was source-oriented and target-oriented. However, the estimation quality decreases for higher values of the target prior. In contrast, KM2 estimates improve in such cases, and KM2 outperforms density ratio based estimation when the target prior is equal 0.8. MLLS estimation combined with DRPU was, in most cases, completely inaccurate. When applied with the nnPU model, MLLS produced reasonable estimates for source priors equal to 0.2 and 0.4. For the target prior 0.4, some estimates even slightly outperformed other methods. Nevertheless, for larger source prior values, MLLS performed poorly even with nnPU, indicating that this method is generally unreliable.

Figure~\ref{fig:gauss_accuracy_ta_05} presents the average accuracies achieved by threshold adaptation methods. Overall, nnPU models obtain better results than DRPU models. In particular, nnPU combined with KM2 or density ratio estimates performs very well. The classical nnPU model without threshold adaptation also performs well and is the only method that maintains good performance when the source prior equals 0.8, proving its robustness. DRPU performs slightly better when combined with the KM2 estimate instead of the density ratio estimator originally used in the DRPU framework. The combination of DRPU and KM2 even achieves slightly improved performance compared to nnPU when the source prior equals 0.2. We also include in this comparison the scenario in which nnPU is trained without knowledge of the training prior, requiring this value to be estimated before training. Denoted as \texttt{nnPU+KM2}, the results are close to nnPU, slightly outperforming it for higher values of the target prior, and yielding slightly lower accuracy for lower values of the target prior. Overall, this method provides results as stable as nnPU, handles scenarios where the source class prior is high quite well, and achieves near best accuracies across experiments. However, in some cases, threshold adaptation methods applied to nnPU perform better.

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_accuracy_ta.png}
%     \caption{Average classification accuracy for threshold adjustment methods on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
%     \label{fig:gauss_accuracy_ta_05}
% \end{figure}

\subsection{Analysis of Threshold Influence on Accuracy}

To further analyze how the selected threshold influences accuracy in the threshold adaptation framework, we calculated the accuracy as a function of the threshold for selected experiments. Figure \ref{fig:gauss_threshold_accuracy} presents four combinations of source and target priors (${(0.2,0.4), (0.4,0.2), (0.4,0.8), (0.8,0.4)}$) aggregated over ten iterations of the experiment. Each graph presents the accuracy as a function of the threshold in the range $[0, 1]$, where each point is the average value over ten runs. We present both nnPU and DRPU models as a basis for threshold adaptation, and they are trained under the assumption of a known source prior. Additionally, we mark four reference threshold values: the standard threshold equal to $0.5$, used in binary classification without any adaptation, and three thresholds calculated from the threshold adaptation formula derived in Section \ref{sec:ta}:
$$
t = \frac{\pi (1 - \hat{\pi}')}{\pi + \hat{\pi}' -2\pi \hat{\pi}'},
$$
where $\hat{\pi}'$ is the estimated target prior using KM2 or density ratio based estimation. These two estimations are aggregated across all ten iterations of experiments $T = \frac{1}{10} \sum_{i=1}^{10}t_i$. We also include the value obtained when $\hat{\pi}'$ is replaced with the true value of the target prior (denoted as \texttt{True}).

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{2. thesis/img/results_img/gauss-avg_accuracy_threshold_0.2-0.4_0.4-0.2_0.4-0.8_0.8-0.4_.png}
%     \caption{Classification accuracy as a function of decision threshold for nnPU and DRPU on synthetic Gaussian data, $n=n'=5000$, $c=0.5$. Vertical lines mark thresholds estimated by KM2 and DRE. For reference, we also show the non-adaptive threshold (0.5) and the threshold computed using the true target prior.}
%     \label{fig:gauss_threshold_accuracy}
% \end{figure}

The figure aims to answer why we do not observe significant improvements in classification performance after applying the threshold adaptation methodology. Indeed, the classification accuracy function becomes almost flat over most of the threshold range. This implies that, for this synthetic dataset, modifying the threshold does not significantly influence classification performance. The standard threshold ($0.5$) appears to provide near-maximal accuracy for many cases. Hence, we do not observe significant improvements from this methodology. A noticeable difference appears when the source prior equals 0.4 and the target prior equals 0.8, where the modified thresholds shift toward the point of maximal accuracy. However, on the contrary, in the reversed scenario, for the nnPU model, we observe a significant drop in accuracy at the point where the perfectly estimated target prior would yield the modified threshold. This confirms the unstable behavior of threshold adaptation observed in the lower-right plot in Figure \ref{fig:gauss_accuracy_ta_05}, where these methods perform very poorly.

\section{Classification Performance for Posterior Adjustment and Target Retraining Methods} \label{sec:res-2-3}

In this section, we present the results for the remaining groups of methods: posterior adjustment and target retraining. Figure~\ref{fig:gauss_accuracy_mlls_05} shows the average accuracies achieved by the MLLS algorithm combined with nnPU and DRPU models. For reference, the results of classical nnPU and DRPU models are also included.

As discussed previously, the MLLS procedure frequently encountered convergence issues, particularly for higher values of the source prior. Therefore, we excluded from the analysis experimental runs in which the MLLS estimate did not satisfy the acceptance criterion $|\hat{\pi'} - \pi'| < 0.15$. The plots indicate the number of accepted runs out of the ten experimental repetitions for each configuration. To avoid underestimating the potential performance of MLLS, aggregation was performed only over the accepted runs.

MLLS combined with DRPU produced accepted results only when the source prior was equal to 0.2 or 0.4. When combined with nnPU, MLLS additionally returned acceptable results for the source prior equal to 0.6. For the source prior equal to 0.8, both models consistently failed to converge to reasonable estimates. Overall, when MLLS successfully converged, its performance was generally comparable to, but not better than, the classical nnPU and DRPU methods. The only configurations in which MLLS achieved higher accuracy occurred when the target prior was set to 0.8.

Figure \ref{fig:gauss_accuracy_target_05} presents the accuracy results for the target retraining method. Compared to standard nnPU, the variant trained on a mixture of the source positive set and the target set demonstrates slightly better classification performance across all configurations. On the other hand, training on target data for the DRPU model does not always provide better results than standard DRPU, as for a source prior equal to 0.2 the classical DRPU performs significantly better. When the source prior is set to 0.4, classical DRPU is still better, but the difference between the two is smaller. For higher values of the source prior, DRPU trained on target data starts outperforming classical DRPU. It is worth noting that the method discussed in this paragraph should intuitively perform better for higher values of the source prior. According to our experimental setup, the target dataset does not change, but with higher values of the source prior the number of labeled positives available for training increases, which should result in improved training and consequently better performance. Indeed, the results across various source priors seem to improve accordingly. Another important factor worth noting is that the experimental setup does not allow for a completely fair comparison between this methodology and the remaining ones, as we modify the training sets in a way that does not exactly replicate the original configurations, i.e., the total number of observations used for training differs. Overall, the standard nnPU procedure applied to a mixture of positives from the source and target appears very promising, yielding the best classification accuracy for the majority of tested configurations.

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_accuracy_mlls.png}
%     \caption{Average classification accuracy for posterior adjustment methods (MLLS) on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
%     \label{fig:gauss_accuracy_mlls_05}
% \end{figure}

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_accuracy_target.png}
%     \caption{Average classification accuracy for target retraining methods on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
%     \label{fig:gauss_accuracy_target_05}
% \end{figure}

\section{Balanced Accuracy Results}

Figures \ref{fig:gauss_balanced_accuracy_ta_05}, \ref{fig:gauss_balanced_accuracy_mlls_05}, and \ref{fig:gauss_balanced_accuracy_target_05} present the results for the three outlined method groups discussed above in terms of accuracy, but here they show balanced accuracy instead. We include these plots as an additional important factor in our analysis. Although we focus on accuracy as our primary metric, it is worth considering balanced accuracy, as we deal with label shift and the data is often imbalanced.

In particular, we observe in some edge cases (when the source prior is equal to 0.8 and the target prior is 0.2) that the balanced accuracy is close to 0.5, indicating that the models classify all data points as a single class. For threshold adaptation methods, this may suggest that the threshold was shifted too aggressively, so that all output scores lie on one side of the threshold. In general, threshold adaptation methods perform well for source priors equal to 0.2 and 0.4, but the metric drops significantly for higher values of the source prior across all threshold adaptation variants.

The standard nnPU model, both with and without the source prior estimating step, similarly to the accuracy results, performs very well and remains stable across all configurations. The MLLS procedure for source priors 0.2 and 0.4 also achieves good results. However, for the nnPU model, which was able to produce some acceptable results under our filtering criterion, we again observe that in the imbalanced case (source prior 0.6 and target prior 0.2) the performance is very poor, achieving an average balanced accuracy of 0.6, while the accuracy for this case was close to 0.85.

Finally, the target retraining methods demonstrate quite good results, reflecting their strong performance observed in the previously discussed sections, with the exception of scenarios where the target prior is equal to 0.2. In these cases, it appears that the majority of data points are again classified as a single class.

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_balanced_accuracy_ta.png}
%     \caption{Average balanced accuracy for threshold adjustment methods on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
%     \label{fig:gauss_balanced_accuracy_ta_05}
% \end{figure}

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_balanced_accuracy_mlls.png}
%     \caption{Average balanced accuracy for posterior adjustment methods (MLLS) on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
%     \label{fig:gauss_balanced_accuracy_mlls_05}
% \end{figure}

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_balanced_accuracy_target.png}
%     \caption{Average balanced accuracy for target retraining methods on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
%     \label{fig:gauss_balanced_accuracy_target_05}
% \end{figure}

\section{Distribution of Modeled Posterior Probability}

In this section, we examine sample distributions of estimated posterior probability by the models throughout our experiments to better understand how the nnPU and DRPU models behave. Figure \ref{fig:model_scores_comparison} presents the distributions of modeled posterior probabilities on the target dataset for nnPU and DRPU from the first iteration of the experiment. For nnPU these values are simply the model scores, and for DRPU, these are model scores multiplied by the target prior estimate, which in this case is set to 0.2.

For the nnPU model, we observe two clusters of points located close to the extreme values, 0 and 1. Their sizes appear to reflect the class proportions. In the range $[0.2, 0.8]$, there are very few points. For DRPU, the first thing we notice is that the values do not lie in the range $[0,1]$, as the posterior values should. Due to the model's nature, they instead lie in $[0, \infty)$. Because of that, in some methods it is necessary to clip the values exceeding 1 to 1. There is also a very high bar at the value 0, indicating that many scores are exactly zero. Regardless of the classification method, these points will be labeled as negatives. These zero scores arise from the DRPU model architecture, which applies the ReLU function to its final outputs:
$$
ReLu(x) = max(0,x).
$$
In mathematical terms, a true density ratio equal to zero would indicate that $p_+(x)=0$ while $p(x)>0$, meaning that the observation lies outside the support of the positive class. In practice, such support mismatch is unlikely to occur so frequently, which raises the question of whether the model's architecture could be improved to better distinguish between truly negligible density ratios and values that are falsely truncated to zero by the activation function.
 
% \begin{figure}[htbp]
%     \centering
%     \subfigure[nnPU model scores]{
%         \includegraphics[width=0.48\linewidth]{2. thesis/img/results_img/gauss_model_scores_nnpu.png}
%         \label{fig:model_scores_nnpu}
%     }
%     \hfill
%     \subfigure[DRPU model scores]{
%         \includegraphics[width=0.48\linewidth]{2. thesis/img/results_img/gauss_model_scores_drpu.png}
%         \label{fig:model_scores_drpu}
%     }
%     \caption{Sample distributions of output  scores (posterior predictions) generated by nnPU and DRPU models for synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $\pi=\pi'=0.2$.}
%     \label{fig:model_scores_comparison}
% \end{figure}

\section{Model errors}

Figure \ref{fig:gauss_erros_grid} presents the distribution of classification errors $| \hat{y}(x_i) - \tilde{y}(x_i) |$, where $\hat{y}(x_i)$ is a true label (1 for positive class and 0 for negative), and $\tilde{y}(x_i)$ is the estimated posterior probability. The distributions are collected from four models: standard nnPU and DRPU (trained only on source data), and their variants trained also on target data. These are results collected from the first iteration of experiment for configuration when the source prior is set to 0.4 and the target prior to 0.6.  

% \begin{figure}[htbp]
% \centering
% \subfigure[nnPU]{
% \includegraphics[width=0.48\linewidth]{2. thesis/img/results_img/gauss_errors_nnpu.png}
% \label{fig:gauss_erros_nnpu}
% }
% \hfill
% \subfigure[DRPU]{
% \includegraphics[width=0.48\linewidth]{2. thesis/img/results_img/gauss_errors_drpu.png}
% \label{fig:gauss_erros_drpu}
% }

% \vspace{0.5em}

% \subfigure[nnPU+Target]{
%     \includegraphics[width=0.48\linewidth]{2. thesis/img/results_img/gauss_errors_nnpu_mixed.png}
%     \label{fig:gauss_erros_nnpu_mixed}
% }
% \hfill
% \subfigure[DRPU+Target]{
%     \includegraphics[width=0.48\linewidth]{2. thesis/img/results_img/gauss_errors_drpu_mixed.png}
%     \label{fig:gauss_erros_drpu_mixed}
% }

% \caption{Classification errors for synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $\pi=0.4$, $\pi'=0.6$.}
% \label{fig:gauss_erros_grid}
% \end{figure}

\section{ROC Curve Analysis}

In this section, we conduct ROC curve analysis. Figures \ref{fig:gauss_roc_curve_nnpu} and \ref{fig:gauss_roc_curve_drpu} present the ROC curves for the nnPU and DRPU models, respectively, for selected pairs of source and target priors, calculated for the first iteration of the experiment. The plots include the key thresholds used in threshold adaptation methods: the standard threshold $0.5$, thresholds obtained when the target prior is estimated using KM2 or a density ratio estimator, and the threshold computed using the true value of the target prior. Additionally, we calculate the Youden statistic $J = \arg\max_{t} (TPR - FPR)$, which corresponds to the threshold maximizing balanced accuracy. The four presented label shift configurations are $(\pi, \pi') = {(0.2,0.4), (0.4,0.2), (0.4,0.8), (0.8,0.4)}$. The precise values of the individual thresholds are displayed in the graph legends.

For the nnPU model, the Youden statistic is often quite close to the standard threshold $0.5$, although its value is always slightly smaller than $0.5$. For DRPU, the Youden statistic is even smaller, typically ranging from $0.05$ to $0.2$. In the majority of cases, the thresholds based on the true prior, KM2, and density ratio estimates lie close to each other. However, in some cases, we observe that the thresholds are shifted substantially toward the region where the false positive rate is close to 0 (e.g., when $(\pi, \pi') = (0.8,0.4)$). For the DRPU model, threshold adaptation methods, in cases where $\pi < \pi'$, tend to align with the Youden statistic, indicating that the balanced accuracy is close to optimal.

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{2. thesis/img/results_img/gauss_0.2-0.4_0.4-0.2_0.4-0.8_0.8-0.4_roc_grid_nnpu.png}
%     \caption{Sample ROC Curve plots based on nnPU model for synthetic Gaussian data, $n=n'=5000$, $c=0.5$.}
%     \label{fig:gauss_roc_curve_nnpu}
% \end{figure}

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{2. thesis/img/results_img/gauss_0.2-0.4_0.4-0.2_0.4-0.8_0.8-0.4_roc_grid_drpu.png}
%     \caption{Sample ROC Curve plots based on DRPU model for synthetic Gaussian data, $n=n'=5000$, $c=0.5$.}
%     \label{fig:gauss_roc_curve_drpu}
% \end{figure}

\section{Performance under Extreme Class Imbalance} % pi, pi' -> 1

In this section, we conduct an additional experiment for a challenging label shift configuration in which both the source and target priors are close to 1. Specifically, the source prior is set to $0.9$, while the target prior is $0.99$. It is important to emphasize that, with the dataset size fixed at $5{,}000$, only a very small number of negative observations is available for training, which makes the task particularly difficult.

Table \ref{tab:gauss_extreme_prior_mae} presents the mean absolute error of target prior estimation, averaged over ten experiment iterations. The MLLS procedure, when combined with either nnPU or DRPU, failed to converge in all runs, therefore, we excluded it from this analysis. Table \ref{tab:gauss_extreme_accuracy} reports the classification accuracy and balanced accuracy, also averaged over ten iterations. Figures \ref{fig:gauss_extreme_prior_mae} and \ref{fig:gauss_extreme_accuracy} illustrate these results in the form of boxplots.

Regarding target prior estimation, both KM2 and density ratio estimation based using model trained on target data produced completely inaccurate estimates. In contrast, the standard density ratio estimation approach was able to provide reliable estimates. Among the evaluated methods, some were also clearly unreliable, for instance those that effectively classified all observations as a single class (\texttt{nnPU+TA+KM2}, \texttt{DRPU+Target}).

The best-performing approaches were the standard nnPU model and its variant with an estimated source prior, as well as nnPU combined with threshold adaptation using a density ratio estimate of the target prior. The standard DRPU model produced some acceptable results, however, its average accuracy remained below 0.6, which can be considered a relatively poor performance.

\begin{table}[htbp]
  \centering
  \small
  \setlength{\tabcolsep}{6pt}
  \begin{tabular}{
      p{4cm}
      p{5cm}
  }
    \toprule
    \textbf{Method} & \textbf{MAE $\pm$ SE} \\
    \midrule
    \texttt{KM2}              & 0.9298 $\pm$ 0.0099 \\
    \texttt{DRE}              & 0.0966 $\pm$ 0.0031 \\
    \texttt{DRE (Target)} & 0.9265 $\pm$ 0.0015 \\
    \bottomrule
  \end{tabular}
  \caption{Mean absolute error of target prior estimation for synthetic Gaussian data under extreme class imbalance ($\pi=0.9$, $\pi'=0.99$, $n=n'=5000$, $c=0.5$, $K=10$).}
  \label{tab:gauss_extreme_prior_mae}
\end{table}

\begin{table}[htbp]
  \centering
  \small
  \setlength{\tabcolsep}{6pt}
  \begin{tabular}{
      p{3.5cm}
      p{4cm}
      p{4cm}
  }
    \toprule
    \textbf{Method} & \textbf{Avg. Acc. $\pm$ SE} & \textbf{Avg. Bal. Acc. $\pm$ SE} \\
    \midrule
    \texttt{nnPU}            & 0.8802 $\pm$ 0.0069 & 0.8583 $\pm$ 0.0037 \\
    \texttt{nnPU+KM2}        & 0.8628 $\pm$ 0.0048 & 0.8683 $\pm$ 0.0023 \\
    \texttt{nnPU+TA+KM2}     & 0.0100 $\pm$ 0.0000 & 0.5000 $\pm$ 0.0000 \\
    \texttt{nnPU+Target}     & 0.0611 $\pm$ 0.0061 & 0.4555 $\pm$ 0.0106 \\
    \texttt{nnPU+TA+DRE}     & 0.8349 $\pm$ 0.0161 & 0.8255 $\pm$ 0.0049 \\
    \midrule
    \texttt{DRPU}            & 0.5846 $\pm$ 0.0194 & 0.6496 $\pm$ 0.0116 \\
    \texttt{DRPU+TA+KM2}     & 0.0868 $\pm$ 0.0029 & 0.5200 $\pm$ 0.0026 \\
    \texttt{DRPU+Target}     & 0.0100 $\pm$ 0.0000 & 0.5000 $\pm$ 0.0000 \\
    \bottomrule
  \end{tabular}
  \caption{Classification performance for synthetic Gaussian data under extreme class imbalance ($\pi=0.9$, $\pi'=0.99$, $n=n'=5000$, $c=0.5$, $K=10$).}
  \label{tab:gauss_extreme_accuracy}
\end{table}

% \begin{figure}
%     \centering
%     \includegraphics[width=0.6\textwidth]{2. thesis/img/results_img/gauss_0.5_boxplot_mae_extreme.png}
%     \caption{Mean absolute errors of target prior estimation for synthetic Gaussian data under extreme class imbalance ($\pi=0.9$, $\pi'=0.99$, $n=n'=5000$, $c=0.5$, $K=10$).}
%     \label{fig:gauss_extreme_prior_mae}
% \end{figure}

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{2. thesis/img/results_img/gauss_0.5_boxplot_accuracy_extreme.png}
%     \caption{Accuracy for synthetic Gaussian data under extreme class imbalance ($\pi=0.9$, $\pi'=0.99$, $n=n'=5000$, $c=0.5$, $K=10$)..}
%     \label{fig:gauss_extreme_accuracy}
% \end{figure}


% \section{Influence of Label Frequency}
% \section{Influence of Dataset Size}



\chapter{Results on Real Datasets}

In this chapter, we present the results for the six real datasets described in Chapter~\ref{chap:datasets} (MNIST, FashionMNIST, ChestXRay, Electricity, CIFAR-10, and SMSSpam). Here, we focus only on a subset of the methods evaluated on synthetic data, namely those that demonstrated the most promising performance in the previous chapter. Specifically, we include the standard nnPU and DRPU methods, both nnPU and DRPU combined with threshold adaptation using the KM2 estimate (\texttt{nnPU+TA+KM2}, \texttt{DRPU+TA+KM2}), and the nnPU model trained on target data (\texttt{nnPU+Target}). These methods achieved the best overall performance on the synthetic Gaussian data. Therefore, we restrict the analysis to them in order to keep it more compact and easier to follow.

For all experiments, we sampled 5,000 observations for both the source and target sets and fixed the label frequency at $c=0.5$ and the source prior at $\pi=0.5$. The target prior was iterated over $\{0.2, 0.4, 0.6, 0.8\}$. All experiments were repeated ten times, with a new samples drawn from the original dataset in every run, and the reported results were aggregated.

\section{Accuracy Results}

Figure~\ref{fig:real_accuracy} presents six plots, one for each dataset, showing the average classification accuracies obtained for all four target priors. Similarly to the synthetic data experiments, the standard nnPU method proves stable performance and often achieves the highest accuracy. In some configurations, the nnPU model trained on target data outperformed the standard nnPU model. However, although this method appeared quite promising based on the synthetic experiments, its performance decreased on several real datasets. It still performed well on FashionMNIST and Electricity, but for the remaining datasets the accuracy dropped significantly, in particular when the target prior was set to 0.8. The nnPU model combined with threshold adaptation using the KM2 estimator was in most cases just below the standard nnPU, being able to sometimes even perform slightly better (ChestXRay). The standard DRPU method yielded the weakest results across most datasets. Combining DRPU with the KM2 estimator improved its performance, but it still wasn't as good as nnPU.

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{2. thesis/img/results_img/real_0.5_accuracy.png}
%     \caption{Average accuracy results on real datasets, $n=n'=5000$, $c=0.5$.}
%     \label{fig:real_accuracy}
% \end{figure}

\section{Prior Estimation}

We evaluated two methods for class prior estimation in these experiments: KM2 and density ratio-based estimation. Figure~\ref{fig:real_mae} presents the average mean absolute error (MAE) aggregated over ten experimental iterations. In most cases, KM2 outperformed the density ratio method, achieving an MAE below 0.05 when the target prior was equal to 0.2 or 0.4. The density ratio method consistently produced more accurate estimates only for the Electricity dataset. However, as the target prior increased, the MAE of the KM2 estimator also increased. The density ratio method consistently produced more accurate estimates only for the Electricity dataset. However, as the target prior increased, the MAE of the KM2 estimator also increased. The density ratio method also appears to provide more accurate estimates for lower values of the target prior. Overall, KM2 appears to perform very well for lower values of the target prior, whereas for higher target priors the density ratio method may sometimes provide more accurate estimates.

% Figure \ref{fig:real_mae}.

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{2. thesis/img/results_img/real_0.5_mae.png}
%     \caption{Mean absolute error of prior estimation on real datasets, $n=n'=5000$, $c=0.5$.}
%     \label{fig:real_mae}
% \end{figure}

\section{Accuracy as Function of Threshold}

Similarly to the synthetic dataset, we also analyzed how the classification accuracy changes as a function of the decision threshold. Figures~\ref{fig:mnist_accuracy_threshold}-\ref{fig:smsspam_accuracy_threshold} present the average accuracy for selected pairs of source and target priors, aggregated over ten experiment iterations. Unfortunately, the results are largely consistent with those obtained on the synthetic data. In the majority of cases, the accuracy curves are relatively flat, with the standard threshold of $0.5$ often yielding performance close to the maximum observed value. Moreover, we do not observe any accuracy peaks around the theoretically optimal threshold values indicated by the green lines. This behavior contrasts with both the theoretical analysis and the illustrative Positive-Negative experiment presented in Section~\ref{sec:pn_data_ex}, where adapting the threshold according to the class prior shift resulted in a significant improvement in accuracy. These plots therefore provide an explanation for the limited effectiveness of the proposed label shift adaptation methods. 

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{2. thesis/img/results_img/mnist_accuracy_threshold_0.5-0.2_0.5-0.4_0.5-0.6_0.5-0.8_.png}
%     \caption{Classification accuracy as a function of decision threshold for nnPU and DRPU on MNIST dataset, $n=n'=5000$, $c=0.5$. Vertical lines mark thresholds estimated by KM2 and DRE. For reference, we also show the non-adaptive threshold (0.5) and the threshold computed using the true target prior.}
%     \label{fig:mnist_accuracy_threshold}
% \end{figure}

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{2. thesis/img/results_img/fashionmnist_accuracy_threshold_0.5-0.2_0.5-0.4_0.5-0.6_0.5-0.8_.png}
%     \caption{Classification accuracy as a function of decision threshold for nnPU and DRPU on FashionMNIST dataset, $n=n'=5000$, $c=0.5$.}
%     \label{fig:fashionmnist_accuracy_threshold}
% \end{figure}

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{2. thesis/img/results_img/chestxray_accuracy_threshold_0.5-0.2_0.5-0.4_0.5-0.6_0.5-0.8_.png}
%     \caption{Classification accuracy as a function of decision threshold for nnPU and DRPU on ChestXRay dataset, $n=n'=5000$, $c=0.5$.}
%     \label{fig:chestxray_accuracy_threshold}
% \end{figure}

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{2. thesis/img/results_img/electricity_accuracy_threshold_0.5-0.2_0.5-0.4_0.5-0.6_0.5-0.8_.png}
%     \caption{Classification accuracy as a function of decision threshold for nnPU and DRPU on Electricity dataset, $n=n'=5000$, $c=0.5$.}
%     \label{fig:electricity_accuracy_threshold}
% \end{figure}

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{2. thesis/img/results_img/cifar10_accuracy_threshold_0.5-0.2_0.5-0.4_0.5-0.6_0.5-0.8_.png}
%     \caption{Classification accuracy as a function of decision threshold for nnPU and DRPU on CIFAR-10 dataset, $n=n'=5000$, $c=0.5$.}
%     \label{fig:cifar10_accuracy_threshold}
% \end{figure}

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{2. thesis/img/results_img/smsspam_accuracy_threshold_0.5-0.2_0.5-0.4_0.5-0.6_0.5-0.8_.png}
%     \caption{Classification accuracy as a function of decision threshold for nnPU and DRPU on SMSSpam dataset, $n=n'=5000$, $c=0.5$.}
%     \label{fig:smsspam_accuracy_threshold}
% \end{figure}

\section{Balanced Accuracy Results}

Figure~\ref{fig:real_balanced_accuracy} presents the average balanced accuracy obtained for all real datasets. As before, the results were aggregated over ten experimental runs. The ranking of the methods is largely consistent with the accuracy results. The standard nnPU model and nnPU combined with threshold adaptation using the KM2 estimator achieve the best performance on most datasets, while nnPU trained on target data generally performs slightly worse. The performance of DRPU is definitely worse then nnPU-based approaches. However, an interesting trend that was not apparent in the accuracy results is that the balanced accuracy of DRPU tends to increase as the target prior becomes larger. 

% \begin{figure}
%     \centering
%     \includegraphics[width=1\textwidth]{2. thesis/img/results_img/real_0.5_balanced_accuracy.png}
%     \caption{Average balanced accuracy results on real datasets, $n=n'=5000$, $c=0.5$.}
%     \label{fig:real_balanced_accuracy}
% \end{figure}

\chapter{Conclusions}

The problem of label shift in Positive Unlabeled learning remains largely unexplored, with only a single publication specifically addressing this topic. While various label-shift adaptation techniques have been proposed in the literature, most of them were developed for fully supervised learning. This study investigates how label shift adaptation methods can be effectively combined with PU learning frameworks and what adjustments are necessary to make them applicable in this setting. The main contributions of this thesis are threshold adaptation methodology and the integration of MLLS procedure into PU learning framework for handling label shift in PU data.

This work investigated label shift adaptation in the case-control PU learning scenario. We evaluated two PU learning frameworks, namely nnPU and DRPU, several class prior estimation methods, and multiple approaches for adapting classifiers to label shift. Within the nnPU framework, both sigmoid and logistic loss functions were considered. In most experimental configurations, the sigmoid loss achieved better classification performance. Nevertheless, it might be beneficial to use nnPU with binary cross-entropy for higher values of target prior like 0.8. Since the target prior is unknown in practice and must first be estimated, choosing the model based on this observation would require training separate models with both loss functions and selecting the appropriate one based on the estimated target prior at given time.

Across nearly all experiments, the nnPU framework consistently outperformed DRPU. An analysis of the posterior predictions generated by both methods suggests that nnPU produces posterior estimates that more closely resemble the true posterior probabilities. In contrast, as shown in Figure \ref{fig:model_scores_comparison} the density ratio estimator learned by DRPU frequently assigned a density ratio equal to zero to a substantial subset of observations, while only a relatively small number of observations obtain scores close to one. Since a zero density ratio implies that an observation lies outside the support of the positive distribution, this behavior appears unrealistic and may partly justify the worse performance of DRPU. One possible explanation of such posterior distribution is the use of the ReLu activation function in the final layer of the density ratio estimator, which forces all negative outputs to zero. Alternative architectural choices or activation functions perhaps might provide more reliable density ratio estimates and improve performance.

When considering label shift adaptation methods, threshold adjustment approaches produced only marginal improvements and often performed similarly to, or worse than, the standard nnPU classifier using the default threshold of 0.5. A comparison of Figures~\ref{fig:thres_adap_ex} and \ref{fig:gauss_threshold_accuracy}, which present synthetic data generated from similar distributions in the Positive-Negative and Positive-Unlabeled settings, respectively, provides a possible explanation for this behavior. In both cases, the same neural network architecture was used. For the PN setting, we observe the expected shape of the accuracy curves, with a clear maximum near the theoretically optimal threshold. In contrast, the corresponding curves obtained in the PU setting are mostly flat. This suggests that the limited effectiveness of threshold adaptation is not caused by the model architecture itself, but rather by the properties of the PU learning framework. Moreover, Figure~\ref{fig:gauss_erros_grid} indicates that a subset of observations receives posterior estimates that are substantially different from their true class labels. These observations are responsible for a large fraction of misclassified observations and are not affected by the moderate threshold modifications. Consequently, changing the classification threshold affects only a small amount of observations and has little influence on the overall classification accuracy. Overall, PU classifiers appear surprisingly robust to prior changes, making label shift adaptation substantially less effective than in standard PN learning.

Posterior adjustment methods based on the MLLS algorithm generally produced very poor results. In many experimental configurations, the algorithm failed to converge to reasonable estimates of the target prior and often resulted in degenerate solutions in which nearly all observations were assigned to a single class. One possible explanation is the use of the Expectation-Maximization algorithm, which does not guarantee convergence to a global optimum and depend on the initialization. In our implementation, the source prior was always used as the initial estimate of the target prior. It is possible that starting the algorithm multiple times with various initial values set from the range $[0, 1]$ could improve the robustness of the procedure and lead to better results.

The target retraining approach, where the model was trained on combination $X_P+X_{U'}$, particularly when combined with the nnPU framework, produced very promising results on synthetic data and often slightly outperformed the standard nnPU model. However, for real world datasets, we observe a small decrease in its performance, placing behind the standard nnPU. Nevertheless, the method may improve the classification accuracy on shifted data, especially in scenarios where a sufficiently large sample from the target distribution is available. 

We evaluated several methods for class prior estimation. The KM2 estimator, which is widely used in the literature, and the density ratio-based estimator employed within the DRPU framework were generally able to produce accurate estimates of the target prior. In the synthetic experiments, the density ratio estimator often outperformed KM2, whereas on the real datasets KM2 typically achieved lower estimation errors. Overall, both methods proved to be reliable. In contrast, the prior estimates obtained through the MLLS procedure were highly unstable due to the convergence issues discussed above. As a result, based on our experiments, MLLS cannot be considered a reliable approach for class prior estimation.

In the experiments, we generally assumed that the source prior was known. However, we also included the \texttt{nnPU+KM2} variant to evaluate a more realistic scenario in which the source prior is unknown and must be estimated from the source data. The comparison between the standard nnPU model and its \texttt{nnPU+KM2} variant indicates that the errors introduced by the additional estimation step have only a minor impact on the final classification performance. Additionally, as already mentioned, both KM2 and the density ratio-based method are generally able to produce accurate estimates of the class prior.

An important limitation of the considered approaches is their reliance on batch processing. Throughout this thesis, all methods were evaluated on batches consisting of 5,000 observations sampled from the target distribution. Consequently, the proposed techniques are not directly applicable to streaming scenarios, where observations arrive continuously, or to situations in which predictions must be made for individual observations. Since class prior estimates must be derived from an entire sample, its accuracy may significantly decrease when only a small amount of target data is available. Furthermore, the target retraining approach requires access to a sufficiently large target sample in order to learn the shifted distribution. Streaming applications with large data volumes could potentially accumulate observations until a batch of sufficient size becomes available, however many use cases require predictions to be generated much more rapidly. Therefore, the area of Positive Unlabeled learning for streaming data, and handling label shift for them is an area for future research.

Overall, the results suggest that adapting PU learning models to label shift is substantially more challenging than adapting fully supervised classifiers. Although class prior estimation can often be performed accurately, correcting for the estimated shift does not necessarily provide improved classification performance.

\chapter{Future Work}

Due to the optimization procedure used by the KM2 estimator, which was highly computational demanding, the experiments conducted in this work were limited to a single synthetic data generator, ten repetitions for each configuration, a fixed label frequency of $c=0.5$, and datasets consisting of $5,000$ observations. Future studies could extend the experimental analysis by considering a broader range of settings. In particular, it would be valuable to investigate the influence of the label frequency, and dataset size on the performance of PU learning methods under label shift.

The MLLS procedure produced unsatisfactory results in the conducted experiments and frequently failed to converge to reasonable estimates. Our analysis identified a potential area for improvement related to the limitations of the Expectation-Maximization algorithm. Future work could investigate alternative initialization strategies, such as multiple runs from different starting values. Furthermore, although this study primarily assumed that the source prior was known, additional experiments could be performed in settings where both the source and target priors must be estimated.

In the experiments, we used a single neural network architecture based on a multilayer perceptron (MLP), following the implementation proposed by \cite{kiryo2017}. Future work could investigate how alternative model architectures perform under the same learning scenario. In particular, it would be valuable to evaluate architectures specifically designed for image data, such as LeNet and convolutional neural networks in the PU under label shift setting.

Finally, as there are not many publicly available datasets that naturally follow the PU learning setting, the label shift simulation procedure proposed in this thesis and described in Algorithm~\ref{alg:shift_data} can be reused in future studies on Positive Unlabeled learning under label shift. The procedure is applicable to binary datasets, so multiclass datasets need to be first binarized. This algorithm is suitable for case-control scenario and may serve as a useful tool for other research on this topic enabling the controlled experiments and facilitating the creation of PU datasets from standard supervised datasets.


\begin{appendices}

\renewcommand{\thefigure}{\thechapter.\arabic{figure}}
\renewcommand{\thetable}{\thechapter.\arabic{table}}
\setcounter{figure}{0}
\setcounter{table}{0}

\chapter{Plots} \label{appendix:plots}

\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_accuracy_loss_ta.png}
    \caption{Comparison of sigmoid (Sig) and binary cross-entropy (CE) losses for threshold adjustment methods on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
    \label{fig:gauss_accuracy_loss_ta_05}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_accuracy_loss_target.png}
    \caption{Comparison of sigmoid (Sig) and binary cross-entropy (CE) losses for MLLS and target retraining methods on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
    \label{fig:gauss_accuracy_loss_target_05}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_mae.png}
    \caption{Mean absolute error of prior estimation on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
    \label{fig:gauss_mae_05}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_accuracy_ta.png}
    \caption{Average classification accuracy for threshold adjustment methods on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
    \label{fig:gauss_accuracy_ta_05}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{2. thesis/img/results_img/gauss-avg_accuracy_threshold_0.2-0.4_0.4-0.2_0.4-0.8_0.8-0.4_.png}
    \caption{Classification accuracy as a function of decision threshold for nnPU and DRPU on synthetic Gaussian data, $n=n'=5000$, $c=0.5$. Vertical lines mark thresholds estimated by KM2 and DRE. For reference, we also show the non-adaptive threshold (0.5) and the threshold computed using the true target prior.}
    \label{fig:gauss_threshold_accuracy}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_accuracy_mlls.png}
    \caption{Average classification accuracy for posterior adjustment methods (MLLS) on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
    \label{fig:gauss_accuracy_mlls_05}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_accuracy_target.png}
    \caption{Average classification accuracy for target retraining methods on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
    \label{fig:gauss_accuracy_target_05}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_balanced_accuracy_ta.png}
    \caption{Average balanced accuracy for threshold adjustment methods on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
    \label{fig:gauss_balanced_accuracy_ta_05}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_balanced_accuracy_mlls.png}
    \caption{Average balanced accuracy for posterior adjustment methods (MLLS) on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
    \label{fig:gauss_balanced_accuracy_mlls_05}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{img/results_img/gauss_0.5_balanced_accuracy_target.png}
    \caption{Average balanced accuracy for target retraining methods on synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $K=10$.}
    \label{fig:gauss_balanced_accuracy_target_05}
\end{figure}

\clearpage

\begin{figure}[htbp!]
    \centering
    \subfigure[nnPU]{
        \includegraphics[width=0.48\linewidth]{2. thesis/img/results_img/gauss_model_scores_nnpu.png}
        \label{fig:model_scores_nnpu}
    }
    \hfill
    \subfigure[DRPU]{
        \includegraphics[width=0.48\linewidth]{2. thesis/img/results_img/gauss_model_scores_drpu.png}
        \label{fig:model_scores_drpu}
    }
    \caption{Sample distributions of posterior probabilities generated by nnPU and DRPU models for synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $\pi=\pi'=0.2$.}
    \label{fig:model_scores_comparison}
\end{figure}

\begin{figure}[htbp!]
\centering
\subfigure[nnPU]{
\includegraphics[width=0.48\linewidth]{2. thesis/img/results_img/gauss_errors_nnpu.png}
\label{fig:gauss_erros_nnpu}
}
\hfill
\subfigure[DRPU]{
\includegraphics[width=0.48\linewidth]{2. thesis/img/results_img/gauss_errors_drpu.png}
\label{fig:gauss_erros_drpu}
}

\vspace{0.5em}

\subfigure[nnPU+Target]{
    \includegraphics[width=0.48\linewidth]{2. thesis/img/results_img/gauss_errors_nnpu_mixed.png}
    \label{fig:gauss_erros_nnpu_mixed}
}
\hfill
\subfigure[DRPU+Target]{
    \includegraphics[width=0.48\linewidth]{2. thesis/img/results_img/gauss_errors_drpu_mixed.png}
    \label{fig:gauss_erros_drpu_mixed}
}

\caption{Classification errors for synthetic Gaussian data, $n=n'=5000$, $c=0.5$, $\pi=0.4$, $\pi'=0.6$.}
\label{fig:gauss_erros_grid}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{2. thesis/img/results_img/gauss_0.2-0.4_0.4-0.2_0.4-0.8_0.8-0.4_roc_grid_nnpu.png}
    \caption{Sample ROC Curve plots based on nnPU model for synthetic Gaussian data, $n=n'=5000$, $c=0.5$.}
    \label{fig:gauss_roc_curve_nnpu}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=1\textwidth]{2. thesis/img/results_img/gauss_0.2-0.4_0.4-0.2_0.4-0.8_0.8-0.4_roc_grid_drpu.png}
    \caption{Sample ROC Curve plots based on DRPU model for synthetic Gaussian data, $n=n'=5000$, $c=0.5$.}
    \label{fig:gauss_roc_curve_drpu}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.6\textwidth]{2. thesis/img/results_img/gauss_0.5_boxplot_mae_extreme.png}
    \caption{Mean absolute errors of target prior estimation for synthetic Gaussian data under extreme class imbalance ($\pi=0.9$, $\pi'=0.99$, $n=n'=5000$, $c=0.5$, $K=10$).}
    \label{fig:gauss_extreme_prior_mae}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=1\textwidth]{2. thesis/img/results_img/gauss_0.5_boxplot_accuracy_extreme.png}
    \caption{Accuracy for synthetic Gaussian data under extreme class imbalance ($\pi=0.9$, $\pi'=0.99$, $n=n'=5000$, $c=0.5$, $K=10$)..}
    \label{fig:gauss_extreme_accuracy}
\end{figure}

% real

\begin{figure}
    \centering
    \includegraphics[width=1\textwidth]{2. thesis/img/results_img/real_0.5_accuracy.png}
    \caption{Average accuracy results on real datasets, $n=n'=5000$, $c=0.5$.}
    \label{fig:real_accuracy}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=1\textwidth]{2. thesis/img/results_img/real_0.5_mae.png}
    \caption{Mean absolute error of prior estimation on real datasets, $n=n'=5000$, $c=0.5$.}
    \label{fig:real_mae}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=1\textwidth]{2. thesis/img/results_img/mnist_accuracy_threshold_0.5-0.2_0.5-0.4_0.5-0.6_0.5-0.8_.png}
    \caption{Classification accuracy as a function of decision threshold for nnPU and DRPU on MNIST dataset, $n=n'=5000$, $c=0.5$. Vertical lines mark thresholds estimated by KM2 and DRE. For reference, we also show the non-adaptive threshold (0.5) and the threshold computed using the true target prior.}
    \label{fig:mnist_accuracy_threshold}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=1\textwidth]{2. thesis/img/results_img/fashionmnist_accuracy_threshold_0.5-0.2_0.5-0.4_0.5-0.6_0.5-0.8_.png}
    \caption{Classification accuracy as a function of decision threshold for nnPU and DRPU on FashionMNIST dataset, $n=n'=5000$, $c=0.5$.}
    \label{fig:fashionmnist_accuracy_threshold}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=1\textwidth]{2. thesis/img/results_img/chestxray_accuracy_threshold_0.5-0.2_0.5-0.4_0.5-0.6_0.5-0.8_.png}
    \caption{Classification accuracy as a function of decision threshold for nnPU and DRPU on ChestXRay dataset, $n=n'=5000$, $c=0.5$.}
    \label{fig:chestxray_accuracy_threshold}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=1\textwidth]{2. thesis/img/results_img/electricity_accuracy_threshold_0.5-0.2_0.5-0.4_0.5-0.6_0.5-0.8_.png}
    \caption{Classification accuracy as a function of decision threshold for nnPU and DRPU on Electricity dataset, $n=n'=5000$, $c=0.5$.}
    \label{fig:electricity_accuracy_threshold}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=1\textwidth]{2. thesis/img/results_img/cifar10_accuracy_threshold_0.5-0.2_0.5-0.4_0.5-0.6_0.5-0.8_.png}
    \caption{Classification accuracy as a function of decision threshold for nnPU and DRPU on CIFAR-10 dataset, $n=n'=5000$, $c=0.5$.}
    \label{fig:cifar10_accuracy_threshold}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=1\textwidth]{2. thesis/img/results_img/smsspam_accuracy_threshold_0.5-0.2_0.5-0.4_0.5-0.6_0.5-0.8_.png}
    \caption{Classification accuracy as a function of decision threshold for nnPU and DRPU on SMSSpam dataset, $n=n'=5000$, $c=0.5$.}
    \label{fig:smsspam_accuracy_threshold}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=1\textwidth]{2. thesis/img/results_img/real_0.5_balanced_accuracy.png}
    \caption{Average balanced accuracy results on real datasets, $n=n'=5000$, $c=0.5$.}
    \label{fig:real_balanced_accuracy}
\end{figure}

\chapter{Technical Documentation} \label{appendix:technical}

This appendix provides a technical description of the PULS (Positive-Unlabeled Learning under Label Shift) experimental framework, including the development environment, code structure, and results collection process. The project codebase is uploaded to GitHub and is available at \url{https://github.com/izabelatelejko/2026-PULS-Master}.

\section{Development Environment} \label{appendix:environment}

The project is developed using Python 3.10 within a Conda environment. GPU acceleration is supported through CUDA 12.8 and PyTorch with CUDA support. The main dependencies are listed in Table~\ref{tab:dependencies}. The environmnet setup process is explained in \texttt{README.md} file in the project repository. 

\begin{table}[htbp]
\centering
\label{tab:dependencies}
\begin{tabular}{ll}
\toprule
\textbf{Package} & \textbf{Version} \\
\midrule
\multicolumn{2}{l}{\textit{Core ML Libraries}} \\
PyTorch & 2.12.0 (nightly) \\
torchvision & 0.26.0 (nightly) \\
NumPy & 1.24.3 \\
Pandas & 2.0.1 \\
scikit-learn & 1.2.2 \\
SciPy & 1.10.1 \\
Transformers & 4.31.0 \\
Sentence-Transformers & 2.2.2 \\
\midrule
\multicolumn{2}{l}{\textit{Datasets and Preprocessing}} \\
Datasets (HuggingFace) & 2.15.0 \\
Pillow & 8.4.0 \\
\midrule
\multicolumn{2}{l}{\textit{Visualization}} \\
Matplotlib & 3.7.1 \\
Seaborn & 0.12.2 \\
\midrule
\multicolumn{2}{l}{\textit{Utilities}} \\
cvxopt & 1.3.2 \\
Pydantic & 2.8.2 \\
pkbar & 0.5 \\
\bottomrule
\end{tabular}
\caption{Project Dependencies}
\end{table}

\section{Code Structure}
\label{appendix:code_structure}

The project is organized as a Python package with three main modules under the \texttt{src/} directory. The project structure is illustrated in Figure~\ref{fig:project_structure}. DRPU module is adapted from \url{https://github.com/csnakajima/pu-learning}, nnPU module is adapted from \url{https://github.com/wawrzenczyka/nnPUss}. The PULS module is the main contribution of this work, implementing experiments for PU learning under Label Shift.

\begin{figure}[htbp]
\hfill
\begin{minipage}{0.85\textwidth}
\dirtree{%
.1 2026-PULS-Master/.
.2 src/.
.3 DRPU/\hspace{8em} \# Density Ratio PU Learning.
.3 nnPU/\hspace{8em} \# Non-negative PU Learning.
.3 PULS/\hspace{8em} \# PU under Label Shift (main module).
.2 data/\hspace{9.4em} \# Datasets (downloaded automatically).
.2 output/\hspace{8.4em} \# Experiment results.
.2 results\_img/\hspace{5.9em} \# Saved plots.
.2 experiments.ipynb\hspace{3.4em} \# Run experiments.
.2 evaluation.ipynb\hspace{3.9em} \# Results analysis and visualization.
.2 examples.ipynb\hspace{4.9em} \# Examples and illustrations.
.2 requirements.txt\hspace{3.9em} \# Python dependencies.
}
\end{minipage}
\caption{Project Directory Structure}
\label{fig:project_structure}
\end{figure}

% \begin{figure}[htbp]
% \dirtree{%
% .1 \texttt{2026-PULS-Master/}.
% .2 \texttt{src/}.
% .3 \texttt{DRPU/} \DTcomment{\texttt{Density Ratio PU Learning}}.
% .3 \texttt{nnPU/} \DTcomment{\texttt{Non-negative PU Learning}}.
% .3 \texttt{PULS/} \DTcomment{\texttt{PU under Label Shift (main module)}}.
% .2 \texttt{data/} \DTcomment{\texttt{Datasets (downloaded automatically)}}.
% .2 \texttt{output/} \DTcomment{\texttt{Experiment results}}.
% .2 \texttt{results\_img/} \DTcomment{\texttt{Saved plots}}.
% .2 \texttt{experiments.ipynb} \DTcomment{\texttt{Run experiments}}.
% .2 \texttt{evaluation.ipynb} \DTcomment{\texttt{Results analysis and visualization}}.
% .2 \texttt{examples.ipynb} \DTcomment{\texttt{Examples and methodology illustrations}}.
% .2 \texttt{requirements.txt} \DTcomment{\texttt{Python dependencies}}.
% }
% \caption{Project Directory Structure}
% \label{fig:project_structure}
% \end{figure}

\subsection{Jupyter Notebooks}
\label{appendix:notebooks}

The project includes three Jupyter notebooks in the root directory for running code:

\begin{description}
    \item[\texttt{experiments.ipynb}] The main notebook for running experiments.
    \item[\texttt{evaluation.ipynb}] Analysis and visualization notebook for aggregating results from multiple experiment runs, generating plots (MAE, accuracy, balanced accuracy, ROC curves).
    \item[\texttt{examples.ipynb}] Examples of Bregman Divergance, PU data, label shift, etc.
\end{description}

\section{Output Structure and Results} \label{appendix:output}

Each experiment results are saved in a hierarchical directory structure under the \texttt{output/} folder. The output path encodes all experimental parameter and follows this structure:

\begin{verbatim}
output/{dataset}/{n}/({mean}/){source_pi}/{target_pi}/{c}/{exp_number}/metrics.json
\end{verbatim}

\noindent where:
\begin{itemize}
    \item \texttt{dataset}: Dataset name (e.g., \texttt{Gauss}, \texttt{MNIST}),
    \item \texttt{n}: Number of samples,
    \item \texttt{mean}: Optional - distribution parameter for synthetic data,
    \item \texttt{source\_pi}: Training set class prior $\pi$,
    \item \texttt{target\_pi}: Test set class prior $\pi'$,
    \item \texttt{c}: Label frequency $c$,
    \item \texttt{exp\_number}: Experiment number.
\end{itemize}

\subsection{Example Path}

For an experiment with:
\begin{itemize}
    \item Gaussian synthetic data,
    \item $n = 5000$ samples,
    \item Distribution mean $= 0.8$,
    \item Source prior $\pi = 0.2$,
    \item Target prior $\pi' = 0.4$,
    \item Label frequency $c = 0.5$,
    \item Experiment iteration $k = 1$.
\end{itemize}

The results are stored at:
\begin{verbatim}
output/Gauss/5000/0.8/0.2/0.4/0.5/1/.
\end{verbatim}

\subsection{Output Structure}

The \texttt{metrics.json} file contains results for all evaluated methods and follows this structure:

\begin{verbatim}
{
  "dataset_stats": {...},
  "test_pis": {
    "true": 0.4,          // True test prior
    "km2": 0.38,          // KM2 estimated prior
    "dre": 0.41,          // DR estimated prior
    "mlls_nnpu": 0.39,    // MLLS estimated prior (nnPU)
    "mlls_drpu": 0.40     // MLLS estimated prior (DRPU)
  },
  "roc_curve": {
    "nnpu": {"fpr": [...], "tpr": [...], "roc_auc": 0.92},
    "drpu": {"fpr": [...], "tpr": [...], "roc_auc": 0.91}
  },
  "accuracy-ta-grid": {    // Threshold adj. on threshold grid
    "nnPU": {
      "accuracy": [...],   // Accuracy values for thresholds in [0,1]
      "thresholds": [...]
    },
    "DRPU": {"accuracy": [...], "thresholds": [...]}
  },
  "nnPU": {"accuracy": 0.85, "f1": 0.84, ...},
  "nnPU+KM2": {...},       // nnPU with estimated prior
  "nnPU+TA+KM2": {...},    // Threshold adj. with KM2
  "nnPU+TA+DRE": {...},    // Threshold adj. with DRE
  "nnPU+MLLS": {...},      // MLLS method
  "nnPU+Target": {...},    // Retrained on target
  "DRPU": {...},
  "DRPU+TA+KM2": {...},
  "DRPU+MLLS": {...},
  "DRPU+Target": {...}
}
\end{verbatim}

\subsection{Loss Functions} \label{appendix:loss}

For given dataset name the results will be stored under two dataset name variants. Under \texttt{dataset\_name} (e.g.\ \texttt{Gauss}) we store DRPU results and nnPU results using sigmoid loss.
Under \texttt{dataset\_name-CE} (e.g.\ \texttt{Gauss-CE}) we store nnPU results using cross-entropy loss.

\section{Azure Deployment} \label{appendix:azure}

Some experiments were deployed to Azure Container Instances to enable parallel execution. The experiment environment is packaged using the provided \texttt{Dockerfile} and pushed to Azure Container Registry. Detailed deployment instructions are provided in \texttt{AZURE\_DEPLOYMENT.md}. Results are downloaded from Azure Blob Storage using the \texttt{download\_azure\_results.py} script.

\section{Reproducibility} \label{appendix:reproducibility}

All experiments use seeded random number generators to ensure reproducibility. The seed is set based on the experiment number (\texttt{exp\_number}), allowing exact reproduction of results when using the same configuration parameters.

\end{appendices}



% ------------------------------- BIBLIOGRAPHY ---------------------------
% LEXICOGRAPHICAL ORDER BY AUTHORS' LAST NAMES
% FOR AMBITIOUS ONES - USE BIBTEX
% \bibliographystyle{abbrv}
\bibliographystyle{agsm}
\bibliography{references}



% ----------------------- LIST OF SYMBOLS AND ABBREVIATIONS ------------------
% \chapter*{List of symbols and abbreviations}

% \begin{tabular}{cl}
% nzw. & nadzwyczajny \\
% * & star operator \\
% $\widetilde{}$ & tilde 
% \end{tabular}
% \\
% If you don't need it, delete it.
% \thispagestyle{empty}


% ----------------------------  LIST OF FIGURES --------------------------------
\listoffigures
\thispagestyle{empty}


% -----------------------------  LIST OF TABLES --------------------------------
\renewcommand{\listtablename}{List of tables}
\listoftables
\thispagestyle{empty}

% -----------------------------  LIST OF APPENDICES ---------------------------
\chapter*{List of appendices}
\addcontentsline{toc}{chapter}{List of appendices}
\noindent A\hspace{2em}Plots \dotfill \pageref{appendix:plots} \\
\noindent B\hspace{2em}Technical Documentation \dotfill \pageref{appendix:technical}
\thispagestyle{empty}


\end{document}
