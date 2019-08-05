package com.Ruksana.SavnnClustering;

import scala.collection.Seq;
import org.apache.spark.sql.expressions.WindowSpec;
import java.util.List;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.api.java.UDF1;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.*;

public class ClusteringArtistsV2 {
//com.Ruksana.SavnnClustering.ClusteringArtistsV2

	
	@SuppressWarnings("rawtypes")
	public static UDF1 udfConvertArrayToVector = new UDF1<Seq<Float>, Vector>() {

		private static final long serialVersionUID = 1L;

		public Vector call(Seq<Float> float_Array) throws Exception {

			List<Float> float_List = scala.collection.JavaConversions.seqAsJavaList(float_Array);
			double[] DoubleArray = new double[float_Array.length()];
			for (int i = 0; i < float_List.size(); i++) {
				DoubleArray[i] = float_List.get(i);
			}
			return Vectors.dense(DoubleArray);
		}
	};
	
	public static void main(String[] args) {
		
			System.out.println("Running the clustering ML model. Version-2....");
			//Setting log levels
			Logger.getLogger("org").setLevel(Level.ERROR);
			Logger.getLogger("akka").setLevel(Level.ERROR);
			
			// WINUTILSconfiguration. Ensure this configuration is present.
			System.setProperty("hadoop.home.dir", "C:\\hadoop\\");
			
			// Create the spark session with configuring required parameters
			SparkSession sparkSession = SparkSession.builder()
					.appName("clustering").master("local[*]")
					.config("spark.driver.memory", "16g")
					.config("spark.executor.memory", "16g")
					.config("spark.executor.instances", 4)
					.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
					.config("spark.kryoserializer.buffer.max.mb", "1024")
	                .config("spark.sql.autoBroadcastJoinThreshold", -1)
	                .config("spark.sql.broadcastTimeout", "36000")
	                .config("maximizeResourceAllocation", true)
	                .config("spark.shuffle.compress", true)
	                .config("spark.shuffle.spill.compress", true)
	                .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:+UnlockDiagnosticVMOptions -XX:+G1SummarizeConcMark -XX:InitiatingHeapOccupancyPercent=35 -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCDateStamps -XX:OnOutOfMemoryError='kill -9 %p'")
	                .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:+UnlockDiagnosticVMOptions -XX:+G1SummarizeConcMark -XX:InitiatingHeapOccupancyPercent=35 -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCDateStamps -XX:OnOutOfMemoryError='kill -9 %p'")
	                .config("spark.dynamicAllocation.enabled", true)
	                .config("spark.shuffle.service.enabled", true)
	                .config("spark.scheduler.mode", "FAIR")
					.getOrCreate();
			
			System.out.println("**********LOADING DATA*************");
			// Loads data
			//Ensure the data is present in the path mentioned. Otherwise this will not load the data.
			
			Dataset<Row> rawDataset = sparkSession.read().option("header", false).option("inferschema", true).csv("spark-warehouse/data/sample100mb.csv");
			Dataset<Row> rawDataset2 = sparkSession.read().option("header", false).option("inferschema", true).csv("spark-warehouse/newmetadata/part*");
			Dataset<Row> rawDataset3 = sparkSession.read().option("header", false).option("inferschema", true).csv("spark-warehouse/notification_clicks/*");
			Dataset<Row> rawDataset4 = sparkSession.read().option("header", false).option("inferschema", true).csv("spark-warehouse/notification_artists/*");
			
			System.out.println("**********DATA CLEANING*************");
			// Separate necessary columns
			Dataset<Row> dataset1 = rawDataset.select(rawDataset.col("_c0").as("userId"), rawDataset.col("_c2").as("songId"));
			Dataset<Row> dataset2 = rawDataset2.select(rawDataset2.col("_c0").as("songId2"),rawDataset2.col("_c1").as("artistId2").cast("Double"));
			Dataset<Row> dataset3 = rawDataset3.select(rawDataset3.col("_c0").as("notificationId3").cast("Double"),rawDataset3.col("_c1").as("userId3"));
			Dataset<Row> dataset4 = rawDataset4.select(rawDataset4.col("_c0").as("notificationId4").cast("Double"),rawDataset4.col("_c1").as("artistId4").cast("Double"));

			// Ignore rows having null values
			Dataset<Row> dataset1Clean = dataset1.na().drop();
			System.out.println("**********dataset1Clean--|userId|  songId|*******");
			Dataset<Row> df2 = dataset2.na().drop();
			System.out.println("**********df2--| songId2| artistId2|*************");
			Dataset<Row> df3 = dataset3.na().drop();
			System.out.println("**********df3--|notoficationId3| userId3|********");
			Dataset<Row> df4 = dataset4.na().drop();
			System.out.println("**********df4--|notoficationId4|artistId4|*******");

			// String indexer on the userId column of data set1
			System.out.println("String indexer on the userId column of data set1");
			StringIndexer userIdIndexer = new StringIndexer().setInputCol("userId").setOutputCol("userIdIndexer");
			StringIndexerModel userIdIndexer_model = userIdIndexer.fit(dataset1Clean);
			Dataset<Row> userIdIndexer_df = userIdIndexer_model.transform(dataset1Clean);
			System.out.println("**********userIdIndexer_df--|userId|songId|userIdIndexer|*************");	
			
			// String indexer on the songId column of data set1
			System.out.println("String indexer on the songId column of data set1");
			StringIndexer songIdIndexer = new StringIndexer().setInputCol("songId").setOutputCol("songIdIndexer");
			StringIndexerModel songIdIndexer_model = songIdIndexer.fit(userIdIndexer_df);
			Dataset<Row> df1 = songIdIndexer_model.transform(userIdIndexer_df);
			System.out.println("**********df1--|userId|songId|userIdIndexer|songIdIndexer|*************");

			//add a column frequency to the dataset 1
			System.out.println("Adding frequency Column to dataset 1");
			Dataset<Row> df1_frequency = df1.groupBy("userIdIndexer", "songIdIndexer").agg(functions.count("*").alias("Frequency"));
			System.out.println("**********df1_frequency--|userIdIndexer|songIdIndexer|Frequency|*******");

			System.out.println("**********ALS model to come up with features************");	
			ALS als = new ALS().setRank(10).setMaxIter(5).setImplicitPrefs(true).setUserCol("userIdIndexer").setItemCol("songIdIndexer").setRatingCol("Frequency");
			ALSModel als_model = als.fit(df1_frequency);
			Dataset<Row> alsFactors = als_model.userFactors();// to get the features out of the dataset passed.
			System.out.println("**********alsFactors--|id|features|*************");
			
			alsFactors.createOrReplaceTempView("SavnnFeatures");
			sparkSession.udf().register("udfConvertArrayToVector", udfConvertArrayToVector, new VectorUDT());
			Dataset<Row> alsFactorsAsVector = sparkSession.sql("SELECT id,udfConvertArrayToVector(features) as featuresvector FROM SavnnFeatures");

			// Scale the variables
			StandardScaler scaler = new StandardScaler().setInputCol("featuresvector").setOutputCol("features").setWithStd(true).setWithMean(true);

			// Compute summary statistics by fitting the StandardScaler
			StandardScalerModel scalerModel = scaler.fit(alsFactorsAsVector);

			// Normalize each feature to have unit standard deviation.
			Dataset<Row> scaledData = scalerModel.transform(alsFactorsAsVector);
			System.out.println("**********scaledData--|id|featuresvector|features|*************");
			System.out.println("**********Clustering by k-means on scaled data*************");
			// Setting K as 300 for K-means algorithm
			KMeans kmeans = new KMeans().setK(300).setSeed(1L);
			KMeansModel modelFinal = kmeans.fit(scaledData);
			Dataset<Row> df1_predictions = modelFinal.transform(scaledData);
			Dataset<Row> df1_final = df1_predictions
									 .join(userIdIndexer_df,df1_predictions.col("id").equalTo(userIdIndexer_df.col("userIdIndexer")), "inner")
									 .drop(userIdIndexer_df.col("userIdIndexer"));
			System.out.println("**df1_final--|id|featuresvector|features|prediction|userId|songId|**");
						
			System.out.println("Join the meatdata anf predictions");
			Dataset<Row> saavnClustersJoin_metadata = df2
								.join(df1_final, df2.col("SongID2").equalTo(df1_final.col("SongID")), "inner")
								.drop(df1_final.col("SongID")).select("SongID2", "ArtistID2", "prediction", "UserID");
			System.out.println("****saavnClustersJoin_metadata--|SongID2|ArtistID2|prediction|UserID|***");
			
			Dataset<Row> calcPopular_artist_Df = saavnClustersJoin_metadata.groupBy("prediction", "ArtistID2").agg(functions.count("*").alias("PopularArtistData"));
			System.out.println("***calcPopular_artist_Df--|prediction|ArtistID2|PopularArtistData|*********");
			
			WindowSpec w = org.apache.spark.sql.expressions.Window.partitionBy("prediction").orderBy(functions.desc("PopularArtistData"));
			Dataset<Row> popularArtist_df= calcPopular_artist_Df.withColumn("rn", row_number().over(w)).where("rn = 1")
											.select(calcPopular_artist_Df.col("prediction"), calcPopular_artist_Df.col("ArtistID2"));
			System.out.println("**********--popularArtist_df-|prediction|ArtistID2|*********");
			

			Dataset<Row> user_cluster_artist_df = df1_final.join(popularArtist_df,df1_final.col("prediction")
												 .equalTo(popularArtist_df.col("prediction")), "inner")
												 .drop(popularArtist_df.col("prediction"))
												 .select("UserID", "prediction","ArtistID2");
			
			System.out.println("**********--user_cluster_artist_df--|UserID|prediction|ArtistID2|******");
			
			System.out.println("Joining the datset obtained so far with column artistID on notification artists to get notification id.");
			Dataset<Row> artistId_notification_prediction_df = df4.join(popularArtist_df,df4.col("artistId4").
													equalTo(popularArtist_df.col("ArtistID2")), "inner")
													.drop(popularArtist_df.col("ArtistID2"));
			System.out.println("***--artistId_notification_prediction_df--|notificationId4|artistId4|prediction|***");
			
			Dataset<Row> userID_Notification_df = artistId_notification_prediction_df.join(saavnClustersJoin_metadata,
									 artistId_notification_prediction_df.col("artistId4").equalTo(saavnClustersJoin_metadata.col("ArtistID2"))
									.and(artistId_notification_prediction_df.col("prediction").equalTo(saavnClustersJoin_metadata.col("prediction"))),
									"inner").drop(artistId_notification_prediction_df.col("prediction"))
									.drop(artistId_notification_prediction_df.col("ArtistID4"))
									.select("UserID", "NotificationID4");
			System.out.println("****--userID_Notification_df--|UserID|NotificationID4|***");
			
			Dataset<Row> userID_artistId_Notification_df = artistId_notification_prediction_df.join(saavnClustersJoin_metadata,
					 artistId_notification_prediction_df.col("artistId4").equalTo(saavnClustersJoin_metadata.col("ArtistID2"))
					.and(artistId_notification_prediction_df.col("prediction").equalTo(saavnClustersJoin_metadata.col("prediction"))),
					"inner").drop(artistId_notification_prediction_df.col("prediction"))
					.drop(artistId_notification_prediction_df.col("ArtistID4"));
					//.select("UserID", "ArtistID2" ,"NotificationID4");
			System.out.println("****--userID_artistId_Notification_df-|UserID|ArtistID2|NotificationID4|***");
		
			//userID_artistId_Notification_df.show();
				
			Dataset<Row> notifications_Sent_by_Model = userID_Notification_df.groupBy("NotificationID4").agg(functions.count("*").alias("notificationsSentCount"));
			System.out.println("***--notifications_Sent_by_Model--|NotificationID4|notificationsSentCount|***");
			
			Dataset<Row> notifications_Clicked_df = userID_Notification_df.join(df3,
													userID_Notification_df.col("UserID").equalTo(df3.col("UserID3"))
													.and(userID_Notification_df.col("NotificationID4").equalTo(df3.col("NotificationID3"))),
													"inner").drop(userID_Notification_df.col("UserID")).
													drop(userID_Notification_df.col("NotificationID4"));
			System.out.println("**********--notifications_Clicked_df--|notificationId3|userId3|*******");
			
			Dataset<Row> notifications_clicked_byUsers = notifications_Clicked_df.groupBy("NotificationID3").agg(functions.count("*").alias("notificationsClickedCount"));
			System.out.println("***--notifications_clicked_byUsers--|notificationId3|userId3|***");
			
			Dataset<Row> notifications_sent_clicked_df = notifications_Sent_by_Model.join(notifications_clicked_byUsers, 
														 notifications_Sent_by_Model.col("NotificationID4").
														 equalTo(notifications_clicked_byUsers.col("NotificationID3")), "inner")
														.drop(notifications_Sent_by_Model.col("NotificationID4"));
			
			System.out.println("***--notifications_sent_clicked_df--|notificationsSentCount|NotificationID3|notificationsClickedCount***");
			
			Dataset<Row> CTR_df = notifications_sent_clicked_df.withColumn("CTR", col("notificationsClickedCount").divide(col("notificationsSentCount")));
			System.out.println("***--CTR_df--|notificationsSentCount|NotificationID3|notificationsClickedCount|CTR|**");
			
			System.out.println("***********Saving the CTR for each of the NotificationIds-point 4 from submissions***************");
			CTR_df.createOrReplaceTempView("CTR_df");
			
			Dataset<Row> CTR_in_percentage = sparkSession.sql("SELECT NotificationID3 as NotificationID, CTR*100 as CTR FROM CTR_df order by CTR desc limit 5");
			CTR_in_percentage.show();
			CTR_in_percentage.coalesce(1).write().option("header", true).mode(SaveMode.Overwrite).option("header", true).format("csv").save("output" + "/"+"csv/CTR/");
			
			
			System.out.println("***********Saving the Cluster Information for the NotificationIds point5 from submissions***************");
			
			userID_artistId_Notification_df.createOrReplaceTempView("userIDArtistId");
			Dataset<Row> csvPoint5 = sparkSession.sql("SELECT * FROM userIDArtistId where NotificationID4 in ('9563.0','9673.0','9692.0','9661.0','9667.0')");
			csvPoint5.createOrReplaceTempView("csvPoint5");
			Dataset<Row> clusterInfo_9563 = sparkSession.sql("SELECT  distinct UserID,ArtistID2 as ArtistID FROM csvPoint5 where NotificationID4 in ('9563.0')");
			clusterInfo_9563.write().option("header", true).parquet("output" + "/" + "parquet/clusterInformation/9563");
			Dataset <Row> clusterInfo_9563_parquet = sparkSession.read().parquet("output" + "/" + "parquet/clusterInformation/9563");
			clusterInfo_9563_parquet.coalesce(1).write().option("header", true).mode(SaveMode.Overwrite).format("csv").save("output" + "/"+"csv/clusterInformation/9563");
			
			Dataset<Row> clusterInfo_9673 = sparkSession.sql("SELECT  distinct UserID,ArtistID2 as ArtistID FROM csvPoint5 where NotificationID4 in ('9673.0')");
			clusterInfo_9673.write().option("header", true).parquet("output" + "/" + "parquet/clusterInformation/9673");
			Dataset <Row> clusterInfo_9673_parquet = sparkSession.read().parquet("output" + "/" + "parquet/clusterInformation/9673");
			clusterInfo_9673_parquet.coalesce(1).write().option("header", true).mode(SaveMode.Overwrite).format("csv").save("output" + "/"+"csv/clusterInformation/9673");
			
			Dataset<Row> clusterInfo_9692 = sparkSession.sql("SELECT  distinct UserID,ArtistID2 as ArtistID  FROM csvPoint5 where NotificationID4 in ('9692.0')");
			clusterInfo_9692.write().option("header", true).parquet("output" + "/" + "parquet/clusterInformation/9692");
			Dataset <Row> clusterInfo_9692_parquet = sparkSession.read().parquet("output" + "/" + "parquet/clusterInformation/9692");
			clusterInfo_9692_parquet.coalesce(1).write().option("header", true).mode(SaveMode.Overwrite).format("csv").save("output" + "/"+"csv/clusterInformation/9692");
			
			Dataset<Row> clusterInfo_9661 = sparkSession.sql("SELECT  distinct UserID,ArtistID2 as ArtistID  FROM csvPoint5 where NotificationID4 in ('9661.0')");
			clusterInfo_9661.write().option("header", true).parquet("output" + "/" + "parquet/clusterInformation/9661");
			Dataset <Row> clusterInfo_9661_parquet = sparkSession.read().parquet("output" + "/" + "parquet/clusterInformation/9661");
			clusterInfo_9661_parquet.coalesce(1).write().option("header", true).mode(SaveMode.Overwrite).format("csv").save("output" + "/"+"csv/clusterInformation/9661");
			
			Dataset<Row> clusterInfo_9667 = sparkSession.sql("SELECT  distinct UserID,ArtistID2 as ArtistID  FROM csvPoint5 where NotificationID4 in ('9667.0')");
			clusterInfo_9667.write().option("header", true).parquet("output" + "/" + "parquet/clusterInformation/9667");
			Dataset <Row> clusterInfo_9667_parquet = sparkSession.read().parquet("output" + "/" + "parquet/clusterInformation/9667");
			clusterInfo_9667_parquet.coalesce(1).write().option("header", true).mode(SaveMode.Overwrite).format("csv").save("output" + "/"+"csv/clusterInformation/9667");
						
			System.out.println("***********Saving the Intermediate Output***************");
			user_cluster_artist_df.createOrReplaceTempView("IntermediateOutput");
			Dataset<Row> IntermediateOutput = sparkSession.sql("SELECT distinct UserId,prediction as ClsuterId,ArtistId2 as ArtistId FROM IntermediateOutput");
			IntermediateOutput.write().option("header", true).parquet("output" + "/" + "parquet/intermediate_output/");
			Dataset <Row> intermediateOutput = sparkSession.read().parquet("output" + "/" + "parquet/intermediate_output/");
			intermediateOutput.coalesce(1).write().option("header", true).mode(SaveMode.Overwrite).option("header", true).format("csv").save("output" + "/"+"csv/intermediate_output/");
			
			sparkSession.stop();
			System.out.println("***********Successfully completed teh clustering model***************");
	}

}
