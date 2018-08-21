using System;

namespace Utils
{

    public static class DateTimeExt
    {
        public static string ToDotDotDot(this DateTime dateTime)
        {
            return string.Format
                (
                    "{0}.{1}.{2}",
                    dateTime.Month.ToString().PadLeft(2, "0".ToCharArray()[0]),
                    dateTime.Day.ToString().PadLeft(2, "0".ToCharArray()[0]),
                    dateTime.Year.ToString().PadLeft(2, "0".ToCharArray()[0])
                );
        }

        public static DateTime RandomTimeAgo(int maxDaysAgo)
        {
            var randy = new Random();
            return DateTime.Now - new TimeSpan(randy.Next(maxDaysAgo - 1), randy.Next(24), randy.Next(60), randy.Next(60));
        }

        public static TimeSpan JustHoursMinutesSeconds(this TimeSpan timeSpan)
        {
            return new TimeSpan(timeSpan.Hours, timeSpan.Minutes, timeSpan.Seconds);
        }

        public static TimeSpan? JustHoursMinutesSeconds(this TimeSpan? timeSpan)
        {
            return timeSpan.HasValue
                ? new TimeSpan(timeSpan.Value.Hours, timeSpan.Value.Minutes, timeSpan.Value.Seconds)
                : (TimeSpan?)null;
        }

        public static string ToDateFileFormat(this DateTime dateTime)
        {
            return String.Format("{0}.{1}.{2}", dateTime.Month, dateTime.Day, dateTime.Year);
        }

        public static string ToDateTimeFileFormat(this DateTime dateTime)
        {
            return String.Format("_{0}-{1}-{2}_{3}-{4}-{5}", dateTime.Month, dateTime.Day, dateTime.Year, dateTime.Hour, dateTime.Minute, dateTime.Second);
        }
    }
}
