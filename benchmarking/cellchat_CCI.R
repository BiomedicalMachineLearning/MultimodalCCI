library(CellChat)
library(Seurat)
library(SeuratDisk)
library(patchwork)
options(stringsAsFactors = FALSE)

# Functions ===============

add_image <- function(data) {
    image <- matrix(1, ncol = (max(obj@meta.data$imagecol) - min(obj@meta.data$imagecol) + 1), nrow = (max(obj@meta.data$imagerow) - min(obj@meta.data$imagerow) + 1))  
    coordinates <- data@meta.data[,c('imagerow','imagecol','imagerow','imagecol')]
    colnames(coordinates) <- c('row','col','imagerow','imagecol')
    coordinates$tissue <- rep(1, nrow(coordinates))
    sfs <- scalefactors(spot = 55, fiducial = 88.84758, hires = 0.0333, lowres = 0.01)
    data@images$slice1 <- new('VisiumV1', 
                              image=image,
                              scale.factors=sfs,
                              coordinates=coordinates,
                              spot.radius=1.6
                             )
    data@images$slice1@assay <- 'Spatial'
    data@images$slice1@key <- 'slice1_'
    return(data)}

update_CellChat_Db <- function(update_LRs_path){
    connectome <- read.csv(update_LRs_path,sep="\t")
    connectome <- connectome[c("Ligand.gene.symbol","Receptor.gene.symbol")]
    colnames(connectome) <- c("ligand","receptor")
    connectome$interaction_name <- paste0(connectome$ligand,"_",connectome$receptor)
    rownames(connectome) <- connectome$interaction_name
    interaction <- CellChatDB.mouse$interaction
    exclude_interactions <- rownames(interaction)
    exclude_interactions <- c(exclude_interactions,"IL36G_IL1RL2_IL1RAP","RARRES2_GPR1")
    interaction <- plyr::rbind.fill(interaction,connectome)
    interaction <- interaction[!duplicated(interaction$interaction_name), ]
    rownames(interaction) <- interaction$interaction_name
    interaction <- interaction[!(rownames(interaction) %in% exclude_interactions), ]
    CellChatDB.mouse.updated <- list()
    CellChatDB.mouse.updated$interaction <- interaction
    CellChatDB.mouse.updated$complex <- CellChatDB.mouse$complex
    CellChatDB.mouse.updated$cofactor <- CellChatDB.mouse$cofactor
    CellChatDB.mouse.updated$geneInfo <- data.frame(Symbol=unique(c(connectome$ligand,connectome$receptor)))
    return(CellChatDB.mouse.updated)}

# Run CellChat ===============

args <- commandArgs(trailingOnly = TRUE)
jobid <- as.numeric(args[1])
samples <- c("sample1", "sample2", "sample3")

obj = readRDS(paste0("/scratch/project/stseq/Onkar/BigData/MMCCI/",samples[jobid],".rds"))
# obj <- SCTransform(obj)
obj <- add_image(obj)
data.input = GetAssayData(obj, slot = "data", assay = "Spatial") 
data.input <- normalizeData(data.input) # normalize data matrix
meta = data.frame(labels = Seurat::Idents(obj), slices = "slice1", row.names = names(Seurat::Idents(obj)))
meta$slices <- factor(meta$slices)
meta$labels <- factor(obj@meta.data$cell_type)
spatial.locs = GetTissueCoordinates(obj, scale = NULL, cols = c("imagerow", "imagecol")) 
spot.size = 55 
sfs <- scalefactors(spot = 55, fiducial = 88.84758, hires = 0.0333, lowres = 0.01)
conversion.factor = spot.size/sfs$spot
spatial.factors = data.frame(ratio = conversion.factor, tol = spot.size/2)


cellchat <- createCellChat(object = data.input, meta = meta, group.by = "labels",
                           datatype = "spatial", coordinates = spatial.locs, spatial.factors = spatial.factors)
CellChatDB.use <- update_CellChat_Db("connectomedb2020.txt")
cellchat@DB <- CellChatDB.use
CellChatDB.mouse <- CellChatDB.use
cellchat <- subsetData(cellchat)
future::plan("multisession", workers = 4) 
cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat)

# Try-1 ==========
cellchat <- computeCommunProb(cellchat, type = "truncatedMean", trim = 0.1, 
                              distance.use = TRUE, interaction.range = 130, scale.distance = 1.5, k.min=8, 
                              contact.knn.k = 8)
cellchat <- filterCommunication(cellchat, min.cells = 8)
cellchat <- aggregateNet(cellchat)
cellchat@dr <- subsetCommunication(cellchat)
write.csv(cellchat@dr, file=paste0("/scratch/project/stseq/Onkar/BigData/MMCCI/cellchat_v1_",samples[jobid],".csv"))


# Try-2 ==========
cellchat2 <- computeCommunProb(cellchat, type = "truncatedMean", trim = 0.1, 
                              distance.use = TRUE, interaction.range = 110, scale.distance = 1.5,
                              contact.knn.k = 8)
cellchat2 <- filterCommunication(cellchat2, min.cells = 8)
cellchat2 <- aggregateNet(cellchat2)
cellchat2@dr <- subsetCommunication(cellchat2)
write.csv(cellchat@dr, file=paste0("/scratch/project/stseq/Onkar/BigData/MMCCI/cellchat_v2_",samples[jobid],".csv"))