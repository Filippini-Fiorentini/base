###############################
##### CORSO DI STATISTICA #####
## per INGEGNERIA MATEMATICA ##
###############################

###############################################
################ LABORATORIO 1 ################
## INTRODUZIONE A R & STATISTICA DESCRITTIVA ##
###############################################


# Per scaricare R:   http://www.r-project.org/
# Per maggiori informazioni su R:
# - dall'interfaccia R: 'aiuto' -> 'Guida Html'
#   oppure 'aiuto' -> 'sito CRAN'

# Argomenti trattati nel laboratorio 1:
# - Comandi base di R (scalari, vettori, matrici e relative operazioni)
# - Import/Export dataframe, Grafici
# - Esempi di analisi descrittiva di variabili qualitative, quantitative;
#   esempi di analisi di un campione e di due campioni

# R: linguaggio interpretato; è possibile:
# - scrivere il codice direttamente sulla console
# - (preferibile) scrivere il codice su uno script (come questo) e poi eseguirlo
#   in console, una riga per volta oppure una selezione. Per eseguire il codice in
#   console: ctrl + r

# Commento: tutto quanto preceduto da '#' non viene letto dalla Console di R
# è possibile dunque eseguire indistintamente nella Console comandi e commenti
# senza dover togliere questi ultimi
# (è SEMPRE opportuno commentare i propri script come mostrato a laboratorio)

# MOLTO IMPORTANTE: come Matlab, R deve avere una DIRECTORY DI LAVORO,
# ovvero una cartella dove di default verranno cercati o salvati i file utilizzati da R
# quindi: create immediatamente una cartella in 'C:/USER_DATA', e chiamatela
# 'nome_cognome_lab1'; salvate poi in questa cartella i file che trovate sul sito del corso
# (pacchetti, dati,..) all'INIZIO di ogni laboratorio

# per selezionare la directory di lavoro:
# seleziono la finestra della Console, e poi bottone 'file' -> 'cambia directory',
# oppure con un comando da tastiera
setwd('C:/USER_DATA/script_Lab1')
# setwd('E:/Didattica/Statistica A (MAT)/laboratori/lab 1')  
# se non mi ricordo la directory di lavoro:
getwd()


### R come calcolatore

# è possibile utilizzare R per eseguire operazioni semplicissime

(17*0.35)^(1/3)

# in R sono definite le principali funzioni matematiche
# (alcune serviranno spesso nell'analisi dei dati!)

log(10)

exp(1)

3^-1

1/3



### OGGETTI IN R: assegnare valore alle variabili

# operatore di assegnamento: <-
# funziona anche '='

## scalari

a <- 1
a
a = 2
a

b <- 3
b

4 -> c
c

a <- b
a
b

## vettori

v <- c(2,3,7,10)
# c() è la funzione che serve a concatenare: un vettore è un insieme di numeri concatenati!
v

# vettori costituiti da sequenze ordinate di numeri:
# è possibile automatizzare la procedura

# posso imporre il passo
u <- seq(0,0.5,by=0.1)
u

# oppure la lunghezza del vettore
length(u)
u <- seq(0,0.5,length=10)
u

# passo negativo significa sequenza decrescente
u1 <- seq(0.5,0,by=-0.1)
u1

# sequenza di passo 1
u2 <- 1:5

# vettori costituiti da ripetizioni di numeri:
# è possibile automatizzare la procedura

w <- rep(1,10)
w

# primo argomento di rep: valore o vettore di valori che voglio ripetere
# secondo argomento di rep: valore o vettore di valori che indicano
# come il primo argomento va ripetuto

w1 <- rep(c(1,2,3,4),3)
w1

# quale sarà la lunghezza di w1?
# ...
length(w1)

# N.B. i comandi rep e seq possono essere utilizzati insieme!

w2 <- rep(1:8,rep(3,8))
w2

w3 <- rep(seq(0,10,length=6),1:6) # Per una descrizione completa del comando seq: help(seq)
w3

w4 <- rep(c(5,9,1,3),c(1,4,2,0))
w4

## matrici

W <- matrix(data = c(1,2,3,4,5,6,7,8,9,10,11,12), nrow = 4, ncol = 3, byrow = F)
W

# oppure
W <- rbind(c(1,5,9),c(2,6,10),c(3,7,11),c(4,8,12))
W

# oppure (più furbo..)
W <- cbind(1:4,5:8,9:12)
W

# Attenzione: in R i vettori non sono matrici n*1 o 1*n!
# fondamentale ricordarselo quando si vuole estrarre un elemento 
# da un vettore

v
dim(v) 
cbind(v)
dim(cbind(v))

## Estrazione di elementi da un vettore o da una matrice

v
v[2]
v[2:3]
v[c(1,3)]
v[-1] # tutto il vettore tranne il primo elemento
v[-length(v)] # tutto il vettore tranne l'ultimo elemento

W
# estrazione di elementi da una matrice
W[2,3]
dim(W)
length(W)
W[2:4,1]
W[4,c(1,3)]

# estrazione di righe o colonne di una matrice
W[3,]
W[,2]

# estrazione di sottomatrici
W[c(1,3,4),2:3]



### OPERAZIONI ALGEBRICHE IN R
# NB: R di default effettua le operazioni componente per componente

# consideriamo i seguenti oggetti
a <- 1
b <- 2
c <- c(2,3,4)
d <- c(10,10,10)
e <- c(1,2,3,4)
f <- 1:6
W # dimensioni 4x3 da prima
Z <- rbind(rep(0,3),1:3,rep(10,3),c(4,7,1))
Z # Z ha le stesse dimensioni di W


# operazioni su scalari e vettori

a+b # scalare + scalare
c+d # vettore + vettore
a*b # scalare * scalare
c*d # vettore * vettore (componente per componente)
c %*% d # vettore * vettore (prodotto interno)
outer( c,d ) # vettore * vettore (prodotto esterno)
c+a # vettore + scalare
c^2 # attenzione: operazioni sono sempre componente per componente
exp(c) # vedi sopra
c(c,d,e) # la funzione c() serve a concatenare qualsiasi tipo di oggetti, anche vettori!

# ATTENZIONE!!!
# cosa succede se compio operazioni componente per componente su vettori di dimensione diversa?
# R può giocare dei bruttissimi scherzi: a volte infatti si viene avvisati con un warning,
# ma in altri casi trovare incongruenze nelle dimensioni è più sottile..

c+e # warning ma NON errore: i due vettori hanno dimensioni diverse ma R calcola comunque la loro somma..
    # come? somma per componente fino all'ultimo elemento del più corto, poi 'ricicla'
    # gli elementi del più corto dall'inizio fino a quando non esaurisce il vettore più lungo
    # quindi: warning perché i due vettori non hanno lunghezze una multipla dell'altra

c+f # f è lungo il doppio di c: R non dà neanche un warning e calcola la somma
    # riciclando gli elementi di c

# operazioni su matrici

Z+W # matrice + matrice (componente per componente)

Z*W # matrice * matrice (componente per componente)


## funzioni che permettono operazioni algebriche in R

sum(c) # somma componenti vettore c
sum(Z) # somma componenti matrice Z (somma tutto!!)

prod(c) # prodotto componenti vettore c
prod(Z) # prodotto componenti matrice Z (come sopra)


V <- t(W) # trasposizione di matrice
          # V è una matrice 3x4
V

V*W # matrice * matrice (componente per componente)

# errore: le matrici hanno dimensioni diverse!

# Moltiplicazione matriciale: anche qui bisogna fare attenzione alle dimensioni..
V %*% W
W %*% V 

W+a # matrice + scalare

W
c
W+c # matrice + vettore ... ATTENZIONE! Recicling "column-major", cioè per colonne

#Calcolo dell'inversa di una matrice (da usare con MOLTA parsimonia)
S = matrix( c(1,2,3,6,4,2,7,8,5), ncol = 3, byrow = T);

S.inv = solve(S)

S.inv %*% S

# N.B.
# Non calcola l'inversa, ma la matrice dei reciproci degli elementi di partenza
S^(-1)

#Infatti controllando il prodotto con S non ottengo l'identità.
S^(-1) %*% S
#Se però calcolo il prodotto elemento per elemento, me ne rendo conto.
S^(-1) * S 

# Solve può essere usato anche per risolvere sistemi lineari (interfaccia a routines LAPACK):
# Dato S x = b, vogliamo trovare x:

b = c(1,1,1);

x = solve( S, b )
x

residuo = b - S %*% x
residuo


# Per visualizzare e cancellare le variabili

ls() #fornisce la lista delle variabili esistenti
rm(a) #rimuove la variabile a
ls()
rm(list=ls()) #rimuove tutte le variabili nel workspace
ls()


### ALTRI OGGETTI IN R

## liste: oggetti costituiti da vari oggetti

scritto <- list (corso = 'Statistica per Ingegneria Matematica',
            data.esame = '04/07/2009',
            num_iscritti = 7,
            num_consegnati = 6,
            numeri_matricola = as.character(c(45020,45679,46789,43126,42345,47568,45674)),
            voti = c(30,19,29,NA,25,26,27)
            )
scritto
# estrazione di un elemento da una lista
scritto$voti
# oppure
scritto[[6]] # infatti ho chiamato 'voti' la sesta componente della lista

# N.B. posso anche scrivere:
scritto[6] 
# ma questo non è un vettore, è ancora una lista..infatti:
scritto[[6]][2] # ok
scritto[6][2] # non trova nulla.. devo invece scrivere:
scritto[6]$voti[2] # ma è sconveniente!

dim(scritto)	# non è ben definito!

## data.frame: oggetti costituiti da vettori di ugual lunghezza, anche di tipo diverso.

# N.B. sembrano matrici ma non lo sono; infatti, i vettori in essi contenuti, se presi per 
#      colonna, per R hanno significato di variabili statistiche (posso associare dei nomi!)

esame <- data.frame(
            matricola = as.character(c(45020,45679,46789,43126,42345,47568,45674)),
            voti_S = c(30,19,29,NA,25,26,27), 
            voti_O = c(3,3,1,NA,3,2,NA), 
            voti_TOT = c(30,22,30,NA,28,28,27))
esame

# agli elementi di un dataframe si accede come a quelli di una lista
# altrimenti R non li può vedere..
voti_S
esame$voti_S
esame[[2]]

# è possibile però fare in modo che anche le variabili contenute in un dataframe risultino visibili
attach(esame) 
voti_S
detach(esame)
voti_S

dim(esame)		# lo vede praticamente come una tabella



### GRAFICI 

## funzione plot: grafici nel piano cartesiano..
## argomenti: ascisse e ordinate dei punti da plottare

x <- c(0,1,2,3)
y <- c(4,5,2,10)
y

plot(x,y)

x <- seq(0,3,by=0.01)
y <- x^2
plot(x,y)

# anziché punti, tracciare una linea che passa per i vari punti
plot(x,y,type='l')


# aggiungere in rosso il vettore z=x^3
z <- x^3
points(x,z,type='l',col='red')

# alternativamente è possibile utilizzare direttamente il comando lines
plot(x,y,type='l')
lines(x,z,col='red')

# inoltre, se so di voler aggiungere al plot un altro vettore, posso regolare
# i valori limite sull'asse delle ascisse/ordinate in modo che entrambi i vettori
# vengano interamente visualizzati
plot(x,y,type='l',xlim=range(x),ylim=range(cbind(y,z)))
lines(x,z,col='red')


# provate il comando demo(graphics): rassegna di possibili grafici
# cliccate sul grafico per passare al successivo
# di volta in volta vedremo i comandi per i grafici che ci interessano

# N.B:
# per chiudere tutte le finestre grafiche:
graphics.off()

### HELP 

# cosa fare quando ci si ricorda nome del comando ma non si ricordano
# il suo scopo oppure i suoi argomenti?
# help(NOMECOMANDO)
# ad esempio:
help(hist)

# quando invece non ci si ricorda nome del comando si può utilizzare
# help.search("KEYWORD")
help.search("histogram")
??histogram

# resituisce un elenco di pacchetti::comandi nel cui help è contenuta la keyword
# specificata. A destra compare anche una brevissima spiegazione del comando.

# oppure 
help.start()

### PACCHETTI

# cosa sono i 'pacchetti' di R?
# per questo laboratorio non è necessario alcun pacchetto; in generale, come dice il nome,
# essi sono degli archivi di file .R (o file programmati in altri linguaggi come C o Fortran),
# e di dati, che permettono di definire particolari tipi di funzioni o di effettuare specifiche
# analisi dei dati.
# Si caricano in R tramite il comando nella Console 'Pacchetti' -> 'Installa Pacchetti...' 
# (accedendo direttamente al sito CRAN se si dispone di una connessione), oppure tramite
# 'Pacchetti' -> 'Installa Pacchetti da file zip locali' se si dispone del pacchetto salvato
# sul proprio pc in formato di archivio zip (soluzione che adotteremo nei prossimi lab).
# A questo punto il pacchetto desiderato è nella propria versione di R.
# è necessario però caricare i pacchetti necessari per l'analisi ogni volta che si accede ad R
# tramite il comando library
# es.
library(MASS) # contiene molti dataset

### SALVARE

# per salvare lo script:
# selezionare la finestra dello script, e cliccare il bottone 'file', 'salva', ...

# per salvare un grafico:
# selezionare la finestra del grafico, e cliccare bottone 'file', 'salva con nome', ...

# per salvare tabelle quali datasets, vettori, matrici, o altri oggetti (esportazione dataframe)
# in un file .txt.
# per esempio:
W <- cbind(1:4,5:8,9:12)
W
write.table(W, 'la-mia-matrice.txt')

rm(W)
W
W <- read.table('la-mia-matrice.txt')
W

# oppure è possibile salvare i dati direttamente come variabili di R, in un file .RData
save(W,file='la-mia-matrice.RData')
# evitate se non strettamente necessario (es. matrici enormi)
# perché i .RData occupano molto più spazio dei .txt

rm(W)
W
load('la-mia-matrice.RData')
W

# ATTENZIONE:
# la tabella o il file .RData verrà salvata nella directory di lavoro che avete selezionato
# (se ne avete selezionata una), altrimenti in quella di default (MAI USARE QUELLA DI DEFAULT!!)

write.table(W, 'la-mia-matrice.txt') # controllate nella vostra directory!

# per salvare area di lavoro:
# selezionare la finestra della Console, cliccare il bottone 'file' -> 'salva area di lavoro', ...
# (anche quando richiesto a chiusura di R -> SCONSIGLIATO in questo caso)



### ANALISI DESCRITTIVA DI UNA VARIABILE ALEATORIA QUALITATIVA (DATI CATEGORICI)

### Le variabili qualitative non possono essere descritte numericamente; è possibile
### solamente trovare la tabella di distribuzione di frequenze per le categorie della 
### variabile, e tracciare grafici (diagrammi a barre e a torta).
### Nei prossimi due esercizi cercheremo di prendere confidenza con questi strumenti.

##########################################
#############   ESERCIZIO 1  #############
##########################################

## creo un vettore di realizzazioni di una variabile categorica

# La funzione factor converte l'argomento (vettore di numeri o caratteri)
# in una serie di realizzazioni di una variabile aleatoria categorica,
# i cui valori possibili sono riportati in Levels.

prov <- c("MI","MI","VA","BG","LO","LO","CR","Alt","CR","MI","Alt","CR","LO","VA","MI","Alt","LO","MI")
prov

prov <- factor(prov, levels=c('MI','LO','BG','CR','VA','Alt'))
prov

provi <- as.factor(prov)
provi

plot(prov) # giocare un po' con le opzioni grafiche!
plot(provi,col='red')
# il comando plot, se applicato ad una variabile creata con 'factor' o 'as.factor'
# capisce da solo che deve fare un grafico di una variabile aleatoria categorica, ovvero un
# grafico a barre       

provASSOLUTE <- table(prov) # tabella delle frequenze assolute
provASSOLUTE
provRELATIVE <- table(prov)/length(prov) #tabella delle frequenze realtive
provRELATIVE

prop.table(provASSOLUTE)
barplot(prop.table(provASSOLUTE),xlab="province",ylab="proporzioni",main="Bar plot delle province")

# Per creare un nuovo device grafico, e tenere aperti più grafici contemporaneamente
 x11() #Linux
# windows() #Solo per Windows
# quartz() #Solo per Mac

pie(provRELATIVE)  # grafico a torta
help(pie)
pie(provRELATIVE,labels=c('MI','LO','BG','CR','VA','Alt'),radius=1,
  col=c('red','orange','yellow','green','lightblue','violet'),main='Grafico a torta Province')
# anche qui si può giocare un po' con le opzioni grafiche per abbellire il grafico



##########################################
#############   ESERCIZIO 2  #############
##########################################

## analisi dei dati categorici
## 'Esposizione ai pesticidi'

pesticide <- c('O','O','J','O','J','O','F','O','F','O','N','F','J','J','F','J','O',
        'J','O','N','C','O','F','O','F','N','N','B','B','O','O','N','B','N','B',
        'C','F','J','M','O','O','F','O','O','J','J','J','O','O','B','M','M','O',
        'O','O','B','M','C','B','F')

pesticide

pesticide <- factor(pesticide, levels=c('J','F','B','M','C','N','O'))
#pesticide <- as.factor( pesticide )
pesticide

# grafico a barre (non serve la tabella di distribuzione di frequenze!)
plot(pesticide)

pestASSOLUTE <- table(pesticide) # tabella delle frequenze assolute
pestASSOLUTE
pestRELATIVE <- table(pesticide)/length(pesticide) #tabella delle frequenze realtive
pestRELATIVE

x11()
#windos()  # serve per creare un nuovo device grafico, e tenere aperti più grafici contemporaneamente
pie(pestRELATIVE )  # grafico a torta


### ANALISI DESCRITTIVA DI UNA VARIABILE ALEATORIA QUANTITATIVA (UNIVARIATA = vive in R) 

### Le variabili quantitative possono essere descritte numericamente, utilizzando opportuni indici
### di posizione e di dispersione, e graficamente, grazie agli istogrammi e ai boxplot.
### Nei prossimi esercizi cercheremo di prendere confidenza con questi strumenti

##########################################
#############   ESERCIZIO 3  #############
##########################################

## analisi dei dati quantitativi contenuti 
## nel file di testo 'vitaminaD.txt'

# prima operazione da fare: copio il file 'vitaminaD.txt' nella directory di lavoro

# importazione di dataset
vitaminaD <- read.table('vitaminaD.txt', sep="\t", header=TRUE)

#L'opzione sep=" " specifica il tipo di separatore di colonna.
#In generale, tipici separatori sono "," "." "\t".

vitaminaD

dim(vitaminaD)

# Come estrarre le labels del data frame:
dimnames(vitaminaD)

rownames(vitaminaD)

colnames(vitaminaD)
names(vitaminaD)

# Rendiamo visibili i campi del data frame:
attach(vitaminaD)

## calcolare i principali indici di posizione e di dispersione 

mean(Vitamina_D) # media

var(Vitamina_D) # varianza campionaria (attenzione: formula incorpora correzione di Bessel)

# N.B. la varianza campionaria è la somma degli scarti quadratici dalla media campionaria,
#      diviso per (n - 1). Non esiste un comando in R per calcolare la varianza della popolazione,
#      ovvero la somma degli scarti quadratici dalla media campionaria, diviso per n. Per ottenere
#      la varianza della popolazione è dunque necessario calcolare:
n <- length(Vitamina_D)
(n-1)/n * var(Vitamina_D) # varianza della popolazione
# oppure, dalla definizione:
mean( (Vitamina_D - mean(Vitamina_D))^2 )

      
sd(Vitamina_D) # deviazione standard campionaria (è la radice quadrata del risultato ottenuto con var)
sqrt((n-1)/n)*sd(Vitamina_D) # deviazione standard della popolazione 

min(Vitamina_D) # minimo

max(Vitamina_D) # massimo

range(Vitamina_D)[2] - range(Vitamina_D)[1] # range
## N.B. la funzione range ritorna un vettore contenente max e min!

median(Vitamina_D) # mediana
# la media è maggiore della mediana: possibile asimmetria verso destra della distribuzione
# e/o presenza di outlier

quantile(Vitamina_D,probs=0.25) # primo quartile

quantile(Vitamina_D,probs=0.50) # secondo quartile
## N.B. coincide con la mediana!

quantile(Vitamina_D,probs=0.75) # terzo quartile

quantile(Vitamina_D,probs=0.75) - quantile(Vitamina_D,probs=0.25) # IQR

quantile(Vitamina_D)
# funzione quantile: riassunto di min, max e quartili


## costruire un istogramma

hist(Vitamina_D) # in ordinata ci sono le frequenze assolute
# dall'istogramma si rileva un'asimmetria destra e la possibile esistenza di potenziali valori estremi

hist(Vitamina_D,prob=TRUE) # in ordinata ci sono le densità

hist(Vitamina_D,prob=TRUE,main='Istogramma Vitamina D',xlab='Concentrazione',ylab='Densità')

# l'argomento breaks serve ad imporre un certo numero di classi: R in questo
# caso ne sceglie 8 in automatico (quindi se scriviamo breaks=8 non cambia niente rispetto a prima!)
hist(Vitamina_D,prob=TRUE,main='Istogramma Vitamina D',
    xlab='Concentrazione',ylab='Densità',breaks=8)

# posso giocare con il numero di classi: non esiste un numero di classi
# 'giusto', la scelta sta alla sensibilità dello statistico

hist(Vitamina_D,prob=TRUE,main='Istogramma Vitamina D',xlab='Concentrazione',ylab='Densità',breaks=15)

# si possono imporre classi di ampiezza diversa
# N.B. in questo caso il grafico ha comunque area 1!
hist(Vitamina_D,prob=TRUE,main='Istogramma Vitamina D',xlab='Concentrazione',ylab='Densità',xlim=c(min(Vitamina_D)-5,max(Vitamina_D)+5),
    breaks=c(min(Vitamina_D),20,26,32,38,44,56,68,max(Vitamina_D)))


## costruire la tabella di distribuzione di frequenze

# per avere la tabella in automatico è possibile utilizzare ancora hist,
# con l'opzione 'plot' impostata su FALSE. La funzione restituisce:
# breaks: estremi delle classi
# counts: frequenze assolute delle classi
# intensities & density: densità associate alle classi
# mids: valori centrali delle classi
# xname: nome della variabile
# equidist: booleano. Le classi hanno tutte la stessa ampiezza?

hist(Vitamina_D,plot=FALSE)

# naturalmente è possibile salvare la tabella in una variabile di R
istogramma <- hist(Vitamina_D,plot=FALSE,breaks=8)
estremiclassi <- istogramma$breaks
estremiclassi

frequenzeassolute <- istogramma$counts
frequenzeassolute

totaleosservazioni <- sum(frequenzeassolute)
totaleosservazioni

frequenzerelative <- (frequenzeassolute)/totaleosservazioni
frequenzerelative

density <- istogramma$density
density



## costruire un boxplot

# in orizzontale
boxplot(Vitamina_D,horizontal=TRUE)

# oppure in verticale
boxplot(Vitamina_D,horizontal=FALSE)

boxplot(Vitamina_D,horizontal=FALSE, main="Boxplot Vitamina D",ylab="Concentrazione",ylim=c(0,90))
# COMMENTO AL BOXPLOT:
# possiamo osservare che il 50% dei valori centrali è compreso nell'intervallo [25.5,47.5]
# il fatto che la linea che rappresenta la mediana sia spostata verso il basso della scatola
# evidenzia un'asimmetria destra nella distribuzione dei dati (la coda destra della distribuzione
# è più estesa di quella sinistra). Inoltre, è possibile notare la presenza di un outlier superiore.


## ricordarsi SEMPRE alla fine di un esercizio di eseguire il detach del dataframe!
## (non è improbabile che dataframe diversi contengano variabili con lo stesso nome..)
detach(vitaminaD)

##########################################
#############   ESERCIZIO 4  #############
##########################################

## Esempio guidato di analisi descrittiva di un campione

## analisi dei dati quantitativi contenuti 
## nel file di testo 'magnesio.txt'

# prima operazione da fare: copio il file 'magnesio.txt' nella directory di lavoro

# importazione di dataset
magnesio <- read.table('magnesio.txt', header=T)
magnesio


dim(magnesio)
dimnames(magnesio)
attach(magnesio)

## indici di posizione e di dispersione 

n <- length(Magnesio)
mean(Magnesio) # media
var(Magnesio) # varianza campionaria
(n-1)/n*var(Magnesio) # varianza della popolazione
sd(Magnesio) # deviazione standard campionaria
sqrt((n-1)/n)*sd(Magnesio) # deviazione standard della popolazione
min(Magnesio) # minimo
max(Magnesio) # massimo
range(Magnesio)[2] - range(magnesio)[1] # range
median(Magnesio) # mediana
# media e mediana coincidono! probabilmente distribuzione simmetrica
Q1 <- quantile(Magnesio,0.25) # primo quartile
Q1
Q3 <- quantile(Magnesio,0.75) # terzo quartile
Q3
Q3 - Q1 # IQR

# media e mediana coincidono: probabilmente distribuzione abbastanza simmetrica

## istogramma

hist(Magnesio,prob=TRUE,main='Istogramma Magnesio',xlab='Concentrazione [mmol/l]',ylab='Densità')
# l'istogramma sembra abbastanza simmetrico: classe modale intervallo (0.75;0.8]

# R ha costruito automaticamente un istogramma a sei classi.. sembrano un po' poche.
# regola euristica: numero di classi circa uguale alla radice quadrata di n, dove n è la dimensione
# del campione; nel nostro caso n=140, dunque sembra ragionevole raddoppiare l'ampiezza delle classi
# e sceglierne 12 (quindi impongo breaks = seq(0.65,0.95,.025))

hist(Magnesio,prob=TRUE,main='Istogramma Magnesio',xlab='Concentrazione [mmol/l]',ylab='Densità',breaks=seq(0.65,0.95,.025))

# si osservi come una ripartizione più fine metta in evidenza una possibile bimodalità della distribuzione

## tabella di distribuzione di frequenze

istogramma <- hist(Magnesio,plot=FALSE,breaks = seq(0.65,0.95,.025))
estremiclassi <- istogramma$breaks
estremiclassi

frequenzeassolute <- istogramma$counts
frequenzeassolute

totaleosservazioni <- sum(frequenzeassolute)
totaleosservazioni

frequenzerelative <- (frequenzeassolute)/totaleosservazioni
frequenzerelative

density <- istogramma$density
density

## boxplot

boxplot(Magnesio,horizontal=FALSE, main="Boxplot Magnesio",ylab="Concentrazione [mmol/l]",ylim=c(0.6,1))
# COMMENTO AL BOXPLOT:
# anche il boxplot indica una marcata simmetria nella distribuzione dei dati

detach(magnesio)


##########################################
#############   ESERCIZIO 5  #############
##########################################

## Esempio guidato di analisi descrittiva di due campioni

## analisi dei dati quantitativi contenuti 
## nel file di testo 'temperatura.txt'

# prima operazione da fare: copio il file 'temperatura.txt' nella directory di lavoro

# importazione di dataset
temp <- read.table('temperatura.txt', header=T)
temp


dim(temp)
dimnames(temp)
attach(temp)

# il dataset contiene 3 variabili: Temperatura, Sesso, e Frequenza Cardiaca.
# ci interessa analizzare la temperatura corporea. considereremo 2 tipi di analisi:
# 1) analisi descrittiva dei dati sulla temperatura corporea indipendentemente dal genere
# 2) analisi descrittiva dei dati sulla temperatura corporea nei due sottocampioni individuati dal genere

## ANALISI 1)
#############

## indici di posizione e di dispersione 

mean(Temperatura) # media
n <- length(Temperatura)
(n-1)/n*var(Temperatura) # varianza nella popolazione
sqrt((n-1)/n)*sd(Temperatura) # deviazione standard nella popolazione
min(Temperatura) # minimo
max(Temperatura) # massimo
range(Temperatura)[2] - range(Temperatura)[1] # range
median(Temperatura) # mediana
# media e mediana coincidono! probabilmente distribuzione simmetrica
Q1 <- quantile(Temperatura,0.25) # primo quartile
Q1
Q3 <- quantile(Temperatura,0.75) # terzo quartile
Q3
Q3 - Q1 # IQR

# dati abbastanza concentrati, non sembrano presentare asimmetrie

## istogramma

hist(Temperatura,prob=TRUE,main='Istogramma Temperatura',xlab='temperatura corporea [gradi F]',ylab='Densità')

# leggera asimmetria sinistra della distribuzione

## tabella di distribuzione di frequenze

istogramma <- hist(Temperatura,plot=FALSE)
estremiclassi <- istogramma$breaks
estremiclassi

frequenzeassolute <- istogramma$counts
frequenzeassolute

totaleosservazioni <- sum(frequenzeassolute)
totaleosservazioni

frequenzerelative <- (frequenzeassolute)/totaleosservazioni
frequenzerelative

density <- istogramma$density
density

## boxplot

boxplot(Temperatura,horizontal=FALSE, main='Boxplot Temperatura',ylab='temperatura corporea [gradi F]',ylim=c(96,101))

## ANALISI 2)
#############

## indici di posizione e di dispersione

# per calcolare gli indici di posizione e dispersione separatamente per ogni sottocampione,
# individuato dal sesso, è comodo utilizzare la funzione tapply: essa applica una certa funzione
# (terzo argomento) a ciascuno dei sottoinsiemi - non vuoti - di valori di una variabile (primo argomento)
# individuati da un fattore di raggruppamento (secondo argomento).
# quindi per noi il primo argomento è la variabile Temperatura, il secondo è la variabile Sesso (categorica!)
# metre il terzo sarà di volta in volta la funzione che dobbiamo applicare per ottenere l'indice cercato
# N.B. attenzione alle funzioni a valori vettoriali (come range) o a più argomenti (come quantile)

tapply(Temperatura,Sesso,mean) # media
tapply(Temperatura,Sesso,var) # varianza
tapply(Temperatura,Sesso,sd) # deviazione standard
tapply(Temperatura,Sesso,min) # minimo
tapply(Temperatura,Sesso,max) # massimo

diff(tapply(Temperatura,Sesso,range)$D) # range donne
diff(tapply(Temperatura,Sesso,range)$U) # range uomini

tapply(Temperatura,Sesso,median) # mediana

Q <- tapply(Temperatura,Sesso,quantile) # quartili

Q1 <- c(Q$D[2],Q$U[2]) # primo quartile [donne,uomini]
Q1
Q3 <- c(Q$D[4],Q$U[4]) # terzo quartile [donne,uomini]
Q3
Q3 - Q1 # IQR

# se la funzione aveva dei propri argomenti, si inseriscono come argomenti di tapply dopo la funzione(quarto,quinto,... argomento)
# per esempio, se volessi il 90_esimo percentile dei due gruppi
# 

Q_90 <- tapply(Temperatura,Sesso,quantile,probs=0.9)
Q_90

# sembra che la temperatura media delle donne sia leggermente superiore a quella degli uomini

## istogramma

par(mfrow=c(1,2)) # serve ad affiancare i due istogrammi
# traccio i due istogrammi uno di fianco all'altro in modo da poter effettuare più facilmente
# un confronto (per poter confrontare i due grafici: stessa scala
# sull'asse delle ascisse, stesse classi e (possibilmente) stessa scala sulle ordinate!)
hist(Temperatura[Sesso=='D'],prob=TRUE,main='Istogramma Temperatura Donne',xlab='temperatura corporea [gradi F]',
   ylab='Densità',col='pink',xlim=range(Temperatura),breaks=seq(96,101,.25), ylim = c(0,1))
hist(Temperatura[Sesso=='U'],prob=TRUE,main='Istogramma Temperatura Uomini',xlab='temperatura corporea [gradi F]',
   ylab='Densità',col='lightblue',xlim=range(Temperatura),breaks=seq(96,101,.25), ylim = c(0,1))

# anche l'istogramma evidenzia una tendenza delle donne ad avere una temperatura corporea più alta rispetto
# agli uomini. Il confronto tra i due istogrammi affiancati è comunque difficoltoso: meglio confrontare i boxplot.

## boxplot

# utilizzo la formula per specificare il fattore di raggruppamento
x11()
#windows()
# con la formula:
boxplot(Temperatura ~ Sesso, data=temp, horizontal=FALSE, main='Boxplot Temperatura',names=c('Donne','Uomini'),col=c('pink','lightblue'),ylab='temperatura corporea [gradi F]',ylim=c(94,102))
# oppure scegliendo una sottoselezione:
boxplot(Temperatura[Sesso=='D'],Temperatura[Sesso=='U'], horizontal=FALSE, main='Boxplot Temperatura',names=c('Donne','Uomini'),col=c('pink','lightblue'),ylab='temperatura corporea [gradi F]',ylim=c(94,102))

# COMMENTO AI BOXPLOT:
# l'analisi dei due boxplot affiancati sembra confermare una tendenza della distribuzione della temperatura
# nelle donne a concentrarsi su valori leggermente superiori rispetto a quella negli uomini.
# considerando il numero di outlier, la dispersione sembra maggiore nelle donne.

detach(temp)

