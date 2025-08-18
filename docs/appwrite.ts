import type { Models } from 'node-appwrite';

export interface PartnersTypeRequest {
    logoRef?: string;
    type?: string;
    statusPositiveColor?: string;
    primaryColor?: string;
    publicCockpitEnabled?: boolean;
    statusNeutralColor?: string;
    friendlyName: string;
    highlightInExplorer?: boolean;
    featureSdg?: boolean;
    excludeFromExplorer?: boolean;
    partnerColor?: string;
    partnerType?: string;
    overviewPostsLimit?: number;
    statusNegativeColor?: string;
    cockpitSubdomainActive?: boolean;
    reportsSubdomainActive?: boolean;
    showNameInHeader?: boolean;
    parentPartnerId?: number;
    name: string;
    deleted?: boolean;
}

export interface MeasuresTypeRequest {
    measureStart?: string;
    status?: string;
    partnerId: string;
    deleted?: boolean;
    description?: string;
    measureEnd?: string;
    sdgs?: string[];
    fullDescription?: string;
    type: string;
    operativeGoal?: string;
    department?: string;
    contactInfo?: string;
    accountNumer?: string;
    costsResponsible?: string;
    costUnit?: number;
    responsiblePreson?: string[];
    parentMeasure?: string[];
    isParent?: boolean;
    title: string;
    budget?: number;
    productArea?: string;
    costUnitCode?: string;
    state?: string;
    priority?: string;
}

export interface PinboardTypeRequest {
    userId: string;
    partnerId: string;
    deleted?: boolean;
    idType?: string;
    position?: number;
    idRef?: string;
}

export interface IndicatorsTypeRequest {
    granularity: string;
    shouldIncrease?: boolean;
    publishedUser?: string;
    isGroup?: boolean;
    deletedDimensionId?: string;
    operationalGoal?: string;
    targetValues?: string;
    grouppedIndicators?: string;
    valuesSource?: string;
    actualValues?: string;
    deleted?: boolean;
    dimensionId?: string;
    description: string;
    publishedDate?: string;
    unit?: string;
    partnerId: string;
    published?: boolean;
    calculation?: string;
    title: string;
    sdgs?: string[];
    dimensionIds?: string[];
    sourceUrl?: string;
}

export interface Measure2indicatorTypeRequest {
    indicatorId: string;
    measureId: string;
}

export interface DimensionsTypeRequest {
    partnerId: string;
    iconRef?: string;
    sustainabilityType?: string;
    deleted?: boolean;
    publishedDate?: string;
    description: string;
    deletedParentDimensionId?: string;
    publishedUser?: string;
    name: string;
    published?: boolean;
    parentDimensionId?: string;
    sdgs?: string[];
    strategicGoal?: string[];
}

export interface PostsTypeRequest {
    partnerId: string;
    title: string;
    externalUrl?: string;
    description?: string;
    measureId?: string;
    postDate?: string;
    dimensionId?: string;
    internalFileRef?: string;
    indicatorId?: string;
    publishedUser?: string;
    postType?: string;
    includeInOverview?: boolean;
    deleted?: boolean;
    published?: boolean;
    publishedDate?: string;
}

export interface ImportedIndicatorsTypeRequest {
    title: string;
    calculation?: string;
    publishedUser?: string;
    deleted?: boolean;
    operationalGoal?: string;
    unit?: string;
    targetValues?: string;
    deletedDimensionId?: string;
    valuesSource?: string;
    actualValues?: string;
    externalId: string;
    description: string;
    published?: boolean;
    granularity: string;
    publishedDate?: string;
    partnerId: string;
    internalId?: string;
    dimensionId?: string;
    shouldIncrease?: boolean;
    externalSource: string;
    sourceUrl?: string;
}

export interface ImportedDimensionsTypeRequest {
    strategicGoal?: string;
    name: string;
    internalId?: string;
    sustainabilityType?: string;
    description: string;
    partnerId: string;
    deleted?: boolean;
    deletedParentDimensionId?: string;
    parentDimensionId?: string;
    publishedDate?: string;
    publishedUser?: string;
    iconRef?: string;
    published?: boolean;
    externalId: string;
    externalSource: string;
}

export interface PartnersTeamsTypeRequest {
    title: string;
    description?: string;
    partnerId: string;
    isParent?: boolean;
    parentGroupIds?: string[];
    deleted?: boolean;
    readerIds?: string[];
    writerIds?: string[];
    managerIds?: string[];
}

export interface CostsResponsibleTypeRequest {
    title: string;
    description?: string;
    isParent: boolean;
    parentIds?: string[];
    partnerId: string;
    deleted?: boolean;
    titleMark: string;
}

export interface CommentsTypeRequest {
    itemType: string;
    itemId: string;
    userId?: string;
    content: string;
    partnerId: string;
    deleted?: boolean;
    pinned?: boolean;
}

export enum RecordsLogChangesTypeRequest_eventType {
    create = "create",
    update = "update",
    delete = "delete",
    archive = "archive",
    restore = "restore",
}

export interface RecordsLogChangesTypeRequest {
    partnerId: string;
    userId: string;
    recordId: string;
    recordType: string;
    changedFields?: string[];
    eventType: RecordsLogChangesTypeRequest_eventType;
}

export interface RecordNotesTypeRequest {
    partnerId: string;
    itemType: string;
    itemId: string;
    content: string;
    deleted?: boolean;
    userId: string;
}

export interface TasksTypeRequest {
    title: string;
    status?: string;
    startDate?: string;
    endDate?: string;
    description?: string;
    createdBy: string;
    itemType?: string;
    itemId?: string;
    parentTasks?: string[];
    directParentTask?: string;
    isPrivate?: boolean;
    partnerId: string;
    deleted?: boolean;
    assignedUserId?: string;
    priority?: string;
}

export interface MeasuresExtendedTypeRequest {
    publishedDate?: string;
    published?: boolean;
    publishedUser?: string;
    indicatorIds?: string[];
    dimensionIds?: string[];
    partnerId: string;
    measureId: string;
    relatedUrls?: string[];
    costs?: string[];
    budgetOrigin?: string[];
    costYearSplit?: string[];
    personalEstimate?: string;
    milestones?: string;
}

export interface PersonalNotesTypeRequest {
    partnerId: string;
    userId: string;
    itemType?: string;
    itemId?: string;
    content: string;
    deleted?: boolean;
}

export interface AcademyFoldersTypeRequest {
    name: string;
    description?: string;
    parentFolder?: string;
    parentFolderTree?: string[];
    directFileCount?: number;
    totalFileCount?: number;
    isPublic?: boolean;
    orderWeight?: number;
}

export enum AcademyContentTypeRequest_level {
    easy = "easy",
    medium = "medium",
    hard = "hard",
    expert = "expert",
}

export interface AcademyContentTypeRequest {
    name: string;
    fileId?: string;
    folderId?: string;
    embedHtml?: string;
    creatorId: string;
    summary?: string;
    descriptionSections?: string[];
    isPublished?: boolean;
    level?: AcademyContentTypeRequest_level;
    previewImage?: string;
    fileDownloadText?: string;
}

export interface CommunicationBoardSettingsTypeRequest {
    partnerId: string;
    jsonSettings?: string;
    lastPublished?: string;
    jsonLiveSettings?: string;
}

export enum StrategiesTypeRequest_type {
    Concept = "Concept",
    Strategy = "Strategy",
}

export interface StrategiesTypeRequest {
    name: string;
    imageId?: string;
    description?: string;
    strategicGoals?: string[];
    type?: StrategiesTypeRequest_type;
    partnerId: string;
    sdgs?: string[];
}

export enum StrategiesRelationsTypeRequest_relationType {
    Dimension = "Dimension",
    Indicator = "Indicator",
    Measure = "Measure",
    Post = "Post",
}

export interface StrategiesRelationsTypeRequest {
    partnerId: string;
    strategyId: string;
    relationType: StrategiesRelationsTypeRequest_relationType;
    relationItemId: string;
    relationSubType?: string;
}

export interface DraftDimensionsTypeRequest {
    iconRef?: string;
    deleted?: boolean;
    description: string;
    publishedDate?: string;
    deletedParentDimensionId?: string;
    published?: boolean;
    name: string;
    parentDimensionId?: string;
    publishedUser?: string;
    partnerId: string;
    sustainabilityType?: string;
    sdgs?: string[];
    strategicGoalTmp?: string[];
    strategicGoal?: string[];
    draftId?: string;
    page?: string;
}

export interface DraftMeasuresTypeRequest {
    deleted?: boolean;
    measureStart?: string;
    partnerId: string;
    publishedUser?: string;
    published?: boolean;
    description?: string;
    measureEnd?: string;
    publishedDate?: string;
    status?: string;
    sdgs?: string[];
    fullDescription?: string;
    xdimensionIds?: string[];
    type: string;
    department?: string;
    contactInfo?: string;
    accountNumer?: string;
    costsResponsible?: string;
    costUnit?: number;
    responsiblePreson?: string[];
    parentMeasure?: string[];
    isParent?: boolean;
    operativeGoal?: string;
    title: string;
    xindicatorIds?: string[];
    budget?: number;
}

export interface DraftTasksTypeRequest {
    title: string;
    partnerId: string;
    createdBy: string;
    status?: string;
    startDate?: string;
    endDate?: string;
    description?: string;
    itemType?: string;
    itemId?: string;
    parentTasks?: string[];
    directParentTask?: string;
    isPrivate?: boolean;
    deleted?: boolean;
    assignedUserId?: string;
    priority?: string;
    draftId?: string;
}

export interface DraftsTypeRequest {
    title?: string;
    deleted?: boolean;
    partnerId?: string;
    draftId: string;
}

export interface DraftIndicatorsTypeRequest {
    publishedUser?: string;
    shouldIncrease?: boolean;
    targetValues?: string;
    published?: boolean;
    operationalGoal?: string;
    deletedDimensionId?: string;
    partnerId: string;
    valuesSource?: string;
    deleted?: boolean;
    description: string;
    dimensionId?: string;
    actualValues?: string;
    unit?: string;
    publishedDate?: string;
    title: string;
    calculation?: string;
    granularity: string;
    sdgs?: string[];
    dimensionIds?: string[];
    isGroup?: boolean;
    grouppedIndicators?: string;
    sourceUrl?: string;
    draftId?: string;
}

export const collections = [
{
  name: "Partners",
  collectionName: "partners-collection",
  fields: ["logoRef","type","statusPositiveColor","primaryColor","publicCockpitEnabled","statusNeutralColor","friendlyName","highlightInExplorer","featureSdg","excludeFromExplorer","partnerColor","partnerType","overviewPostsLimit","statusNegativeColor","cockpitSubdomainActive","reportsSubdomainActive","showNameInHeader","parentPartnerId","name","deleted"]
},
{
  name: "Measures",
  collectionName: "measures-collection",
  fields: ["measureStart","status","partnerId","deleted","description","measureEnd","sdgs","fullDescription","type","operativeGoal","department","contactInfo","accountNumer","costsResponsible","costUnit","responsiblePreson","parentMeasure","isParent","title","budget","productArea","costUnitCode","state","priority"]
},
{
  name: "Pinboard",
  collectionName: "pinboard-collection",
  fields: ["userId","partnerId","deleted","idType","position","idRef"]
},
{
  name: "Indicators",
  collectionName: "indicators-collection",
  fields: ["granularity","shouldIncrease","publishedUser","isGroup","deletedDimensionId","operationalGoal","targetValues","grouppedIndicators","valuesSource","actualValues","deleted","dimensionId","description","publishedDate","unit","partnerId","published","calculation","title","sdgs","dimensionIds","sourceUrl"]
},
{
  name: "Measure2indicator",
  collectionName: "measure2indicator",
  fields: ["indicatorId","measureId"]
},
{
  name: "Dimensions",
  collectionName: "dimensions-collection",
  fields: ["partnerId","iconRef","sustainabilityType","deleted","publishedDate","description","deletedParentDimensionId","publishedUser","name","published","parentDimensionId","sdgs","strategicGoal"]
},
{
  name: "Posts",
  collectionName: "posts-collection",
  fields: ["partnerId","title","externalUrl","description","measureId","postDate","dimensionId","internalFileRef","indicatorId","publishedUser","postType","includeInOverview","deleted","published","publishedDate"]
},
{
  name: "ImportedIndicators",
  collectionName: "imported-indicators-collection",
  fields: ["title","calculation","publishedUser","deleted","operationalGoal","unit","targetValues","deletedDimensionId","valuesSource","actualValues","externalId","description","published","granularity","publishedDate","partnerId","internalId","dimensionId","shouldIncrease","externalSource","sourceUrl"]
},
{
  name: "ImportedDimensions",
  collectionName: "imported-dimensions-collection",
  fields: ["strategicGoal","name","internalId","sustainabilityType","description","partnerId","deleted","deletedParentDimensionId","parentDimensionId","publishedDate","publishedUser","iconRef","published","externalId","externalSource"]
},
{
  name: "PartnersTeams",
  collectionName: "partnersTeams",
  fields: ["title","description","partnerId","isParent","parentGroupIds","deleted","readerIds","writerIds","managerIds"]
},
{
  name: "CostsResponsible",
  collectionName: "costs-responsible",
  fields: ["title","description","isParent","parentIds","partnerId","deleted","titleMark"]
},
{
  name: "Comments",
  collectionName: "comments-collection",
  fields: ["itemType","itemId","userId","content","partnerId","deleted","pinned"]
},
{
  name: "RecordsLogChanges",
  collectionName: "records-log-changes",
  fields: ["partnerId","userId","recordId","recordType","changedFields","eventType"]
},
{
  name: "RecordNotes",
  collectionName: "record-notes-collection",
  fields: ["partnerId","itemType","itemId","content","deleted","userId"]
},
{
  name: "Tasks",
  collectionName: "tasks-collection",
  fields: ["title","status","startDate","endDate","description","createdBy","itemType","itemId","parentTasks","directParentTask","isPrivate","partnerId","deleted","assignedUserId","priority"]
},
{
  name: "MeasuresExtended",
  collectionName: "measures-extended-collection",
  fields: ["publishedDate","published","publishedUser","indicatorIds","dimensionIds","partnerId","measureId","relatedUrls","costs","budgetOrigin","costYearSplit","personalEstimate","milestones"]
},
{
  name: "PersonalNotes",
  collectionName: "personal-notes",
  fields: ["partnerId","userId","itemType","itemId","content","deleted"]
},
{
  name: "AcademyFolders",
  collectionName: "academy-folders",
  fields: ["name","description","parentFolder","parentFolderTree","directFileCount","totalFileCount","isPublic","orderWeight"]
},
{
  name: "AcademyContent",
  collectionName: "academy-content",
  fields: ["name","fileId","folderId","embedHtml","creatorId","summary","descriptionSections","isPublished","level","previewImage","fileDownloadText"]
},
{
  name: "CommunicationBoardSettings",
  collectionName: "communication-board-settings",
  fields: ["partnerId","jsonSettings","lastPublished","jsonLiveSettings"]
},
{
  name: "Strategies",
  collectionName: "strategies-collection",
  fields: ["name","imageId","description","strategicGoals","type","partnerId","sdgs"]
},
{
  name: "StrategiesRelations",
  collectionName: "strategies-relations",
  fields: ["partnerId","strategyId","relationType","relationItemId","relationSubType"]
},
{
  name: "DraftDimensions",
  collectionName: "draft-dimensions",
  fields: ["iconRef","deleted","description","publishedDate","deletedParentDimensionId","published","name","parentDimensionId","publishedUser","partnerId","sustainabilityType","sdgs","strategicGoalTmp","strategicGoal","draftId","page"]
},
{
  name: "DraftMeasures",
  collectionName: "draft-measures",
  fields: ["deleted","measureStart","partnerId","publishedUser","published","description","measureEnd","publishedDate","status","sdgs","fullDescription","xdimensionIds","type","department","contactInfo","accountNumer","costsResponsible","costUnit","responsiblePreson","parentMeasure","isParent","operativeGoal","title","xindicatorIds","budget"]
},
{
  name: "DraftTasks",
  collectionName: "draft-tasks",
  fields: ["title","partnerId","createdBy","status","startDate","endDate","description","itemType","itemId","parentTasks","directParentTask","isPrivate","deleted","assignedUserId","priority","draftId"]
},
{
  name: "Drafts",
  collectionName: "drafts",
  fields: ["title","deleted","partnerId","draftId"]
},
{
  name: "DraftIndicators",
  collectionName: "draft-indicators",
  fields: ["publishedUser","shouldIncrease","targetValues","published","operationalGoal","deletedDimensionId","partnerId","valuesSource","deleted","description","dimensionId","actualValues","unit","publishedDate","title","calculation","granularity","sdgs","dimensionIds","isGroup","grouppedIndicators","sourceUrl","draftId"]
},
];

export type CollectionTypesRequests = {
  Partners: PartnersTypeRequest;
  Measures: MeasuresTypeRequest;
  Pinboard: PinboardTypeRequest;
  Indicators: IndicatorsTypeRequest;
  Measure2indicator: Measure2indicatorTypeRequest;
  Dimensions: DimensionsTypeRequest;
  Posts: PostsTypeRequest;
  ImportedIndicators: ImportedIndicatorsTypeRequest;
  ImportedDimensions: ImportedDimensionsTypeRequest;
  PartnersTeams: PartnersTeamsTypeRequest;
  CostsResponsible: CostsResponsibleTypeRequest;
  Comments: CommentsTypeRequest;
  RecordsLogChanges: RecordsLogChangesTypeRequest;
  RecordNotes: RecordNotesTypeRequest;
  Tasks: TasksTypeRequest;
  MeasuresExtended: MeasuresExtendedTypeRequest;
  PersonalNotes: PersonalNotesTypeRequest;
  AcademyFolders: AcademyFoldersTypeRequest;
  AcademyContent: AcademyContentTypeRequest;
  CommunicationBoardSettings: CommunicationBoardSettingsTypeRequest;
  Strategies: StrategiesTypeRequest;
  StrategiesRelations: StrategiesRelationsTypeRequest;
  DraftDimensions: DraftDimensionsTypeRequest;
  DraftMeasures: DraftMeasuresTypeRequest;
  DraftTasks: DraftTasksTypeRequest;
  Drafts: DraftsTypeRequest;
  DraftIndicators: DraftIndicatorsTypeRequest;
};


export type CollectionTypes = {
  Partners: PartnersType;
  Measures: MeasuresType;
  Pinboard: PinboardType;
  Indicators: IndicatorsType;
  Measure2indicator: Measure2indicatorType;
  Dimensions: DimensionsType;
  Posts: PostsType;
  ImportedIndicators: ImportedIndicatorsType;
  ImportedDimensions: ImportedDimensionsType;
  PartnersTeams: PartnersTeamsType;
  CostsResponsible: CostsResponsibleType;
  Comments: CommentsType;
  RecordsLogChanges: RecordsLogChangesType;
  RecordNotes: RecordNotesType;
  Tasks: TasksType;
  MeasuresExtended: MeasuresExtendedType;
  PersonalNotes: PersonalNotesType;
  AcademyFolders: AcademyFoldersType;
  AcademyContent: AcademyContentType;
  CommunicationBoardSettings: CommunicationBoardSettingsType;
  Strategies: StrategiesType;
  StrategiesRelations: StrategiesRelationsType;
  DraftDimensions: DraftDimensionsType;
  DraftMeasures: DraftMeasuresType;
  DraftTasks: DraftTasksType;
  Drafts: DraftsType;
  DraftIndicators: DraftIndicatorsType;
};

export interface PartnersType extends PartnersTypeRequest, Models.Document {}
export interface MeasuresType extends MeasuresTypeRequest, Models.Document {}
export interface PinboardType extends PinboardTypeRequest, Models.Document {}
export interface IndicatorsType extends IndicatorsTypeRequest, Models.Document {}
export interface Measure2indicatorType extends Measure2indicatorTypeRequest, Models.Document {}
export interface DimensionsType extends DimensionsTypeRequest, Models.Document {}
export interface PostsType extends PostsTypeRequest, Models.Document {}
export interface ImportedIndicatorsType extends ImportedIndicatorsTypeRequest, Models.Document {}
export interface ImportedDimensionsType extends ImportedDimensionsTypeRequest, Models.Document {}
export interface PartnersTeamsType extends PartnersTeamsTypeRequest, Models.Document {}
export interface CostsResponsibleType extends CostsResponsibleTypeRequest, Models.Document {}
export interface CommentsType extends CommentsTypeRequest, Models.Document {}
export interface RecordsLogChangesType extends RecordsLogChangesTypeRequest, Models.Document {}
export interface RecordNotesType extends RecordNotesTypeRequest, Models.Document {}
export interface TasksType extends TasksTypeRequest, Models.Document {}
export interface MeasuresExtendedType extends MeasuresExtendedTypeRequest, Models.Document {}
export interface PersonalNotesType extends PersonalNotesTypeRequest, Models.Document {}
export interface AcademyFoldersType extends AcademyFoldersTypeRequest, Models.Document {}
export interface AcademyContentType extends AcademyContentTypeRequest, Models.Document {}
export interface CommunicationBoardSettingsType extends CommunicationBoardSettingsTypeRequest, Models.Document {}
export interface StrategiesType extends StrategiesTypeRequest, Models.Document {}
export interface StrategiesRelationsType extends StrategiesRelationsTypeRequest, Models.Document {}
export interface DraftDimensionsType extends DraftDimensionsTypeRequest, Models.Document {}
export interface DraftMeasuresType extends DraftMeasuresTypeRequest, Models.Document {}
export interface DraftTasksType extends DraftTasksTypeRequest, Models.Document {}
export interface DraftsType extends DraftsTypeRequest, Models.Document {}
export interface DraftIndicatorsType extends DraftIndicatorsTypeRequest, Models.Document {}
