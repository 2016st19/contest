import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

import java.io.IOException;
import java.util.StringTokenizer;

/**
 * Created by jyb on 2/23/17.
 */
public class Join {
    public static class JoinMapper
            extends Mapper<Object, Text, Text, Text> {
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while(itr.hasMoreTokens()){
                context.write(new Text(itr.nextToken().toString()), new Text(itr.nextToken().toString()));
            }
        }
    }

    public static class JoinNewPartitioner
            extends HashPartitioner<Text, Text> {
        public int getPartition(Text key, Text value, int numReduceTasks){
            return super.getPartition(key, value, numReduceTasks);
        }
    }

    public static class JoinReducer
            extends Reducer<Text, Text, NullWritable, Text> {
        private Text word = new Text();
        static Text CurrentItem = new Text("*");
        private static int attrNum = 22;
        private static double[] attrDouble = new double[attrNum];
        private static int[] attrInt = new int[attrNum];
        private static String timeStamp1, timeStamp2;

        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException{
            word.set(key);
            if((!CurrentItem.equals(word)) && (!CurrentItem.equals("*"))) {
                myOutPut(context);
                Initial();
            }
            CurrentItem = new Text(word);
            for(Text v : values) {
                String filename = v.toString().split("#")[0];
                String[] items = v.toString().split("#")[1].split(",");
                if(filename.equals("bill")) {
                    for(int i = 0; i <= 13; ++i){
                        if(i == 0) timeStamp1 = items[i];
                        else if((i == 1) || (i == 7) || (i == 13)) attrInt[i] = Integer.parseInt(items[i]);
                        else attrDouble[i] = Double.parseDouble(items[i]);
                    }
                } else {
                    if(filename.equals("loan")) {
                        timeStamp2 = items[0];
                    } else {
                        int offset = 0;
                        if (filename.equals("browse")) offset = 14;
                        else if (filename.equals("user")) offset = 16;
                        for (int i = 0; i < items.length; ++i) attrInt[i + offset] = Integer.parseInt(items[i]);
                    }
                }
            }
        }

        protected static void Initial() {
            for(int i = 0; i < attrNum; ++i) {
                attrDouble[i] = 0;
                attrInt[i] = 0;
            }
            timeStamp1 = timeStamp2 = "";
        }

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Initial();
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException{
            myOutPut(context);
        }

        private static void myOutPut(Context reducerContext) throws IOException, InterruptedException {
            String output1 = "";
            for(int i = 0; i < attrNum; ++i) {
                if(i == (attrNum - 1)) output1 += timeStamp2;
                else {
                    if(i == 0) output1 += timeStamp1 + ",";
                    else if((i == 1) || (i == 7) || ((13 <= i) && (i <= 20))) output1 += attrInt[i] + ",";
                    else output1 += attrDouble[i] + ",";
                }
            }
            reducerContext.write(NullWritable.get(), new Text(CurrentItem.toString() + "," + output1));
        }
    }

    public static void main(String inputPath, String outBufPath) throws Exception{
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Join");
        job.setNumReduceTasks(1);
        job.setJarByClass(Join.class);
        job.setMapperClass(JoinMapper.class);
        job.setPartitionerClass(JoinNewPartitioner.class);
        job.setReducerClass(JoinReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outBufPath));
        int exit = job.waitForCompletion(true) ? 0 : 1;
    }
}
